extern crate clap;
use clap::{App, Arg, SubCommand};
use log::{debug, error, info, trace, warn};
use simplelog::*;
use sleighcraft::ffi::PcodeOpCode;
use sleighcraft::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use std::convert::TryFrom;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::targets::{InitializationConfig, Target};
use inkwell::types::IntType;
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, CallableValue};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

const REGS_SIZE: usize = 0x400;
#[repr(C)]
struct Emu {
    pub regs: *mut u8,
}

extern "C" fn call_print() -> bool {
    println!("fasdf");
    true
}

type EmuFunc = unsafe extern "C" fn(*mut Emu);

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

struct FuncContext<'ctx> {
    func: FunctionValue<'ctx>,
    uniques: BTreeMap<usize, BasicValueEnum<'ctx>>,
    emu_ptr: BasicValueEnum<'ctx>,
    regs_ptr: BasicValueEnum<'ctx>,
}

impl<'ctx> FuncContext<'ctx> {
    fn new(
        fn_val: FunctionValue<'ctx>,
        emu_ptr: BasicValueEnum<'ctx>,
        regs_ptr: BasicValueEnum<'ctx>,
    ) -> Self {
        Self {
            func: fn_val,
            uniques: BTreeMap::<usize, BasicValueEnum<'ctx>>::new(),
            emu_ptr: emu_ptr,
            regs_ptr: regs_ptr,
        }
    }
}

impl<'ctx> CodeGen<'ctx> {
    fn int_type_from_length(&self, size: u32) -> IntType<'ctx> {
        match size {
            1 => self.context.i8_type(),
            2 => self.context.i16_type(),
            4 => self.context.i32_type(),
            8 => self.context.i64_type(),
            _ => panic!("Can't handle int of size {}", size),
        }
    }

    fn get_sign_bit(&self, val: IntValue<'ctx>) -> IntValue<'ctx> {
        let t = val.get_type();
        let width = t.get_bit_width();
        let sign_bit = self.builder.build_right_shift(
            val,
            t.const_int(width as u64 - 1, false),
            false,
            "v_sign_bit",
        );
        self.builder
            .build_int_cast(sign_bit, self.context.i8_type(), "b_sign_bit")
    }

    fn varnode_read_value(
        &self,
        fctx: &FuncContext<'ctx>,
        node: &PcodeVarnodeData,
    ) -> BasicValueEnum<'ctx> {
        match node.space.as_str() {
            "register" => {
                assert!((node.offset + node.size as usize) < REGS_SIZE);
                let reg_type = self.int_type_from_length(node.size);
                let reg_address = unsafe {
                    self.builder.build_gep(
                        fctx.regs_ptr.into_pointer_value(),
                        &[self.context.i64_type().const_int(node.offset as u64, false)],
                        "reg_address",
                    )
                };
                let ra = self.builder.build_pointer_cast(
                    reg_address,
                    reg_type.ptr_type(AddressSpace::Generic),
                    "reg_addr",
                );
                self.builder.build_load(ra, "reg_val")
            }
            "unique" => fctx.uniques[&node.offset],
            "const" => {
                let t = self.int_type_from_length(node.size);
                BasicValueEnum::IntValue(t.const_int(node.offset as u64, false))
            }
            _ => panic!("Unhandled space \"{}\"", node.space),
        }
    }

    fn varnode_write_value(
        &self,
        fctx: &mut FuncContext<'ctx>,
        node: &PcodeVarnodeData,
        val: BasicValueEnum<'ctx>,
    ) {
        match node.space.as_str() {
            "register" => {
                assert!((node.offset + node.size as usize) < REGS_SIZE);
                let reg_address = unsafe {
                    self.builder.build_gep(
                        fctx.regs_ptr.into_pointer_value(),
                        &[self.context.i64_type().const_int(node.offset as u64, false)],
                        "reg_address",
                    )
                };
                self.builder.build_store(reg_address, val);
            }
            "unique" => {
                fctx.uniques.insert(node.offset, val);
            }
            _ => panic!("Unhandled space \"{}\"", node.space),
        }
    }

    fn jit_compile_pcode(
        &self,
        addr: u64,
        pcode_ops: &[PcodeInstruction],
    ) -> Option<JitFunction<EmuFunc>> {
        // Create our function for this block of pcode
        let fn_type = self.context.void_type().fn_type(
            &[self
                .context
                .i8_type()
                .ptr_type(AddressSpace::Generic)
                .ptr_type(AddressSpace::Generic)
                .into()],
            false,
        );

        let fn_name = format!("func_{:016x}", addr);
        let fn_val = self.module.add_function(&fn_name, fn_type, None);
        let entry_bb = self.context.append_basic_block(fn_val, "entry");
        self.builder.position_at_end(entry_bb);

        let emu_ptr = fn_val.get_first_param().unwrap().into_pointer_value();
        let regs_ptr = self.builder.build_load(emu_ptr, "regs");
        let mut fn_ctx = FuncContext::new(
            fn_val,
            BasicValueEnum::PointerValue(emu_ptr),
            regs_ptr,
        );

        // Create a list of basic blocks correlating to each pcode inst in the slice
        //  to allow for easier branching (internal branching)
        let mut ops_bb = Vec::new();
        for op in pcode_ops {
            let bb = self.context.append_basic_block(fn_ctx.func, "op");
            ops_bb.push((op, bb));
        }
        let ret_block = self.context.append_basic_block(fn_ctx.func, "return");
        
        // Make sure the entry basic block goes to the next block
        let entry_next_bb = entry_bb.get_next_basic_block();
        if let Some(nbb) = entry_next_bb {
            self.builder.build_unconditional_branch(nbb);
        }

        let mut i = 0;
        for (op, bb) in &ops_bb {
            self.builder.position_at_end(*bb);
            debug!("{:?}", op);
            match op.opcode {
                PcodeOpCode::BRANCH | PcodeOpCode::CALL => {
                    // Raw pcode should always have exactly one input for this
                    assert_eq!(op.vars.len(), 1);
                    assert!(op.out_var.is_none());
                    let inp0 = &op.vars[0];
                    match inp0.space.as_str() {
                        "const" => {
                            debug!("Hit internal branch code!");
                            let index = inp0.offset.wrapping_add(i);
                            debug!("Branching internally to pcode inst at {}", index);
                            let (_, target_bb) = ops_bb[index];
                            self.builder.build_unconditional_branch(target_bb);
                        }
                        "ram" => {
                            debug!("Hit external branching code!");
                            let mut found = false;
                            for (t_op, t_bb) in &ops_bb {
                                if t_op.addr.offset == inp0.offset as u64 {
                                    debug!(
                                        "Branching to instruction {:?} @ 0x{:x}",
                                        t_op.opcode, t_op.addr.offset
                                    );
                                    self.builder.build_unconditional_branch(*t_bb);
                                    found = true;
                                    break;
                                }
                            }
                            if !found {
                                panic!("Unimplemented branching outside of known space");
                            }
                        }
                        _ => panic!("Unimplemented branch for space \"{}\"", inp0.space),
                    }
                }
                PcodeOpCode::CBRANCH => {
                    // Raw pcode should always have exactly one input for this
                    assert_eq!(op.vars.len(), 2);
                    assert!(op.out_var.is_none());
                    let inp0 = &op.vars[0];
                    let inp1 = self
                        .varnode_read_value(&fn_ctx, &op.vars[1])
                        .into_int_value();
                    match inp0.space.as_str() {
                        "const" => {
                            debug!("Hit internal branch code!");
                            let index = inp0.offset.wrapping_add(i);
                            debug!("Branching internally to pcode inst at {}", index);
                            let (_, target_bb) = ops_bb[index];
                            self.builder.build_conditional_branch(
                                inp1,
                                target_bb,
                                bb.get_next_basic_block().unwrap(),
                            );
                        }
                        "ram" => {
                            debug!("Hit external branching code!");
                            let mut found = false;
                            for (t_op, t_bb) in &ops_bb {
                                if t_op.addr.offset == inp0.offset as u64 {
                                    debug!(
                                        "Branching to instruction {:?} @ 0x{:x}",
                                        t_op.opcode, t_op.addr.offset
                                    );
                                    self.builder.build_conditional_branch(
                                        inp1,
                                        *t_bb,
                                        bb.get_next_basic_block().unwrap(),
                                    );
                                    found = true;
                                    break;
                                }
                            }
                            if !found {
                                panic!("Unimplemented branching outside of known space");
                            }
                        }
                        _ => panic!("Unimplemented cbranch for space \"{}\"", inp0.space),
                    }
                }
                PcodeOpCode::COPY => {
                    assert_eq!(op.vars.len(), 1);
                    let outvar = match &op.out_var {
                        Some(o) => o,
                        None => panic!("out_var for {:?} must not be None!", op.opcode),
                    };
                    assert_eq!(outvar.size, op.vars[0].size);
                    let inp0 = self.varnode_read_value(&fn_ctx, &op.vars[0]);
                    self.varnode_write_value(&mut fn_ctx, outvar, inp0);
                    let func_ptr: extern fn() -> bool = call_print;
                    let static_func = func_ptr as u64;
                    let ptr_type = self.context.bool_type().fn_type(&[], false).ptr_type(AddressSpace::Generic);
                    let ptr = self.builder.build_int_to_ptr(
                        self.context.i64_type().const_int(static_func, false),
                        ptr_type,
                        "call_func"
                    );
                    debug!("Putting call to {:x}", static_func);
                    self.builder.build_call(CallableValue::try_from(ptr).unwrap(), &[], "call_val");
                }
                /*
                 * INT_SBORROW | INT_SCARRY
                 * Parameters 	Description
                 * input0 		First varnode input.
                 * input1 		Varnode to subtract from first.
                 * output 		Boolean result containing signed overflow condition.
                 */
                PcodeOpCode::INT_SBORROW | PcodeOpCode::INT_SCARRY => {
                    assert_eq!(op.vars.len(), 2);
                    let outvar = match &op.out_var {
                        Some(o) => o,
                        None => panic!("out_var for {:?} must not be None!", op.opcode),
                    };
                    assert_eq!(outvar.size, 1);
                    assert_eq!(op.vars[0].size, op.vars[1].size);

                    let inp0 = self
                        .varnode_read_value(&fn_ctx, &op.vars[0])
                        .into_int_value();
                    let inp1 = self
                        .varnode_read_value(&fn_ctx, &op.vars[1])
                        .into_int_value();
                    let cres = match op.opcode {
                        PcodeOpCode::INT_SBORROW => {
                            self.builder.build_int_sub(inp0, inp1, "sborrow_sub_val")
                        }
                        PcodeOpCode::INT_SCARRY => {
                            self.builder.build_int_add(inp0, inp1, "scarry_add_val")
                        }
                        _ => panic!("Unreachable!"),
                    };

                    let sb0 = self.get_sign_bit(cres);
                    let sb1 = self.get_sign_bit(inp0);
                    let sb2 = self.get_sign_bit(inp1);
                    let sb3 = sb0.get_type().const_int(1, false);
                    let res = match op.opcode {
                        PcodeOpCode::INT_SBORROW => {
                            let a = self.builder.build_xor(sb1, sb0, "a");
                            let r0 = self.builder.build_xor(sb0, sb2, "r0");
                            let r1 = self.builder.build_xor(r0, sb3, "r1");
                            self.builder.build_and(a, r1, "res")
                        }
                        PcodeOpCode::INT_SCARRY => {
                            let r = self.builder.build_xor(sb0, sb1, "r");
                            let a0 = self.builder.build_xor(sb1, sb2, "a0");
                            let a1 = self.builder.build_xor(a0, sb3, "a1");
                            self.builder.build_and(r, a1, "res")
                        }
                        _ => panic!("Unreachable"),
                    };

                    self.varnode_write_value(
                        &mut fn_ctx,
                        outvar,
                        BasicValueEnum::IntValue(res),
                    );
                }
                PcodeOpCode::INT_CARRY => {
                    assert_eq!(op.vars.len(), 2);
                    let outvar = match &op.out_var {
                        Some(o) => o,
                        None => panic!("out_var for {:?} must not be None!", op.opcode),
                    };
                    assert_eq!(outvar.size, 1);
                    assert_eq!(op.vars[0].size, op.vars[1].size);

                    let inp0 = self
                        .varnode_read_value(&fn_ctx, &op.vars[0])
                        .into_int_value();
                    let inp1 = self
                        .varnode_read_value(&fn_ctx, &op.vars[1])
                        .into_int_value();

                    let add = self.builder.build_int_add(inp0, inp1, "sum_val");
                    let masks: [u64; 9] = [
                        0,
                        0xff,
                        0xffff,
                        0xffffff,
                        0xffffffff,
                        0xffffffffff,
                        0xffffffffffff,
                        0xffffffffffffff,
                        0xffffffffffffffff,
                    ];
                    let masked = self.builder.build_and(
                        add,
                        inp0.get_type()
                            .const_int(masks[std::cmp::min(op.vars[0].size as usize, 8)], false),
                        "masked",
                    );
                    let carry =
                        self.builder
                            .build_int_compare(IntPredicate::UGT, inp0, masked, "carry");
                    let i8_type = self.context.i8_type();
                    let res = self.builder.build_and(
                        self.builder.build_int_cast(carry, i8_type, "i8_cmp_res"),
                        i8_type.const_int(1, false),
                        "res",
                    );
                    self.varnode_write_value(
                        &mut fn_ctx,
                        outvar,
                        BasicValueEnum::IntValue(res),
                    );
                }
                /*
                 * All INT_xxx two value same size result operations
                 * Parameters 	Description
                 * input0 		First varnode input.
                 * input1 		Varnode to subtract from first.
                 * output 		Varnode containing result of integer subtraction.
                 */
                PcodeOpCode::INT_ADD
                | PcodeOpCode::INT_SUB
                | PcodeOpCode::INT_XOR
                | PcodeOpCode::INT_AND
                | PcodeOpCode::INT_OR
                | PcodeOpCode::INT_LEFT
                | PcodeOpCode::INT_RIGHT
                | PcodeOpCode::INT_SRIGHT
                | PcodeOpCode::INT_MULT
                | PcodeOpCode::INT_DIV
                | PcodeOpCode::INT_REM
                | PcodeOpCode::INT_SDIV
                | PcodeOpCode::INT_SREM => {
                    assert_eq!(op.vars.len(), 2);
                    let outvar = match &op.out_var {
                        Some(o) => o,
                        None => panic!("out_var for {:?} must not be None!", op.opcode),
                    };
                    assert_eq!(op.vars[0].size, op.vars[1].size);
                    assert_eq!(op.vars[0].size, outvar.size);
                    let inp0 = self
                        .varnode_read_value(&fn_ctx, &op.vars[0])
                        .into_int_value();
                    let inp1 = self
                        .varnode_read_value(&fn_ctx, &op.vars[1])
                        .into_int_value();
                    let res = match op.opcode {
                        PcodeOpCode::INT_ADD => self.builder.build_int_add(inp0, inp1, "res_val"),
                        PcodeOpCode::INT_SUB => self.builder.build_int_sub(inp0, inp1, "res_val"),
                        PcodeOpCode::INT_XOR => self.builder.build_xor(inp0, inp1, "res_val"),
                        PcodeOpCode::INT_AND => self.builder.build_and(inp0, inp1, "res_val"),
                        PcodeOpCode::INT_OR => self.builder.build_or(inp0, inp1, "res_val"),
                        PcodeOpCode::INT_LEFT => {
                            self.builder.build_left_shift(inp0, inp1, "res_val")
                        }
                        PcodeOpCode::INT_RIGHT | PcodeOpCode::INT_SRIGHT => {
                            self.builder.build_right_shift(
                                inp0,
                                inp1,
                                op.opcode == PcodeOpCode::INT_SRIGHT,
                                "res_val",
                            )
                        }
                        PcodeOpCode::INT_MULT => self.builder.build_int_mul(inp0, inp1, "res_val"),
                        PcodeOpCode::INT_DIV => {
                            self.builder.build_int_unsigned_div(inp0, inp1, "res_val")
                        }
                        PcodeOpCode::INT_REM => {
                            self.builder.build_int_unsigned_rem(inp0, inp1, "res_val")
                        }
                        PcodeOpCode::INT_SDIV => {
                            self.builder.build_int_signed_div(inp0, inp1, "res_val")
                        }
                        PcodeOpCode::INT_SREM => {
                            self.builder.build_int_signed_rem(inp0, inp1, "res_val")
                        }
                        _ => panic!("Unreachable!"),
                    };
                    self.varnode_write_value(
                        &mut fn_ctx,
                        outvar,
                        BasicValueEnum::IntValue(res),
                    );
                }
                PcodeOpCode::INT_ZEXT | PcodeOpCode::INT_SEXT => {
                    assert_eq!(op.vars.len(), 1);
                    let outvar = match &op.out_var {
                        Some(o) => o,
                        None => panic!("out_var for {:?} must not be None!", op.opcode),
                    };
                    assert!(op.vars[0].size <= outvar.size);
                    let inp0 = self
                        .varnode_read_value(&fn_ctx, &op.vars[0])
                        .into_int_value();
                    let out_type = self.int_type_from_length(outvar.size);
                    let res = match op.opcode {
                        PcodeOpCode::INT_ZEXT => {
                            self.builder.build_int_z_extend(inp0, out_type, "zext")
                        }
                        PcodeOpCode::INT_SEXT => {
                            self.builder.build_int_s_extend(inp0, out_type, "sext")
                        }
                        _ => panic!("Unreachable!"),
                    };
                    self.varnode_write_value(
                        &mut fn_ctx,
                        outvar,
                        BasicValueEnum::IntValue(res),
                    );
                }
                PcodeOpCode::INT_2COMP | PcodeOpCode::INT_NEGATE => {
                    assert_eq!(op.vars.len(), 1);
                    let outvar = match &op.out_var {
                        Some(o) => o,
                        None => panic!("out_var for {:?} must not be None!", op.opcode),
                    };
                    assert_eq!(op.vars[0].size, outvar.size);
                    let inp0 = self
                        .varnode_read_value(&fn_ctx, &op.vars[0])
                        .into_int_value();
                    let res = match op.opcode {
                        PcodeOpCode::INT_2COMP => self.builder.build_int_neg(inp0, "neg"),
                        PcodeOpCode::INT_NEGATE => self.builder.build_not(inp0, "not"),
                        _ => panic!("Unreachable!"),
                    };
                    self.varnode_write_value(
                        &mut fn_ctx,
                        outvar,
                        BasicValueEnum::IntValue(res),
                    );
                }
                /*
                 * All INT_xxx comparison operations
                 * Parameters 	Description
                 * input0 		First signed varnode to compare.
                 * input1 		Second signed varnode to compare.
                 * output 		Boolean varnode containing result of comparison.
                 */
                PcodeOpCode::INT_EQUAL
                | PcodeOpCode::INT_NOTEQUAL
                | PcodeOpCode::INT_LESS
                | PcodeOpCode::INT_SLESS
                | PcodeOpCode::INT_LESSEQUAL
                | PcodeOpCode::INT_SLESSEQUAL => {
                    assert_eq!(op.vars.len(), 2);
                    let outvar = match &op.out_var {
                        Some(o) => o,
                        None => panic!("out_var for {:?} must not be None!", op.opcode),
                    };
                    assert_eq!(op.vars[0].size, op.vars[1].size);
                    assert_eq!(outvar.size, 1);
                    let inp0 = self
                        .varnode_read_value(&fn_ctx, &op.vars[0])
                        .into_int_value();
                    let inp1 = self
                        .varnode_read_value(&fn_ctx, &op.vars[1])
                        .into_int_value();
                    let cmp_res = self.builder.build_int_compare(
                        match op.opcode {
                            PcodeOpCode::INT_EQUAL => IntPredicate::EQ,
                            PcodeOpCode::INT_NOTEQUAL => IntPredicate::NE,
                            PcodeOpCode::INT_LESS => IntPredicate::ULT,
                            PcodeOpCode::INT_SLESS => IntPredicate::SLT,
                            PcodeOpCode::INT_LESSEQUAL => IntPredicate::ULE,
                            PcodeOpCode::INT_SLESSEQUAL => IntPredicate::SLE,
                            _ => panic!("Shouldn't ever be hit!"),
                        },
                        inp0,
                        inp1,
                        "cmp_res",
                    );
                    let i8_type = self.context.i8_type();
                    let res = self.builder.build_and(
                        self.builder.build_int_cast(cmp_res, i8_type, "i8_cmp_res"),
                        i8_type.const_int(1, false),
                        "res",
                    );
                    self.varnode_write_value(
                        &mut fn_ctx,
                        outvar,
                        BasicValueEnum::IntValue(res),
                    );
                }
                /*
                 * POPCOUNT nonsense
                 * val = (val & 0x5555555555555555L) + ((val >>> 1) & 0x5555555555555555L);
                 * val = (val & 0x3333333333333333L) + ((val >>> 2) & 0x3333333333333333L);
                 * val = (val & 0x0f0f0f0f0f0f0f0fL) + ((val >>> 4) & 0x0f0f0f0f0f0f0f0fL);
                 * val = (val & 0x00ff00ff00ff00ffL) + ((val >>> 8) & 0x00ff00ff00ff00ffL);
                 * val = (val & 0x0000ffff0000ffffL) + ((val >>> 16) & 0x0000ffff0000ffffL);
                 * int res = (int) (val & 0xff);
                 * res += (int) ((val >> 32) & 0xff);
                 * return res;
                 */
                PcodeOpCode::POPCOUNT => {
                    assert_eq!(op.vars.len(), 1);
                    let outvar = match &op.out_var {
                        Some(o) => o,
                        None => panic!("out_var for {:?} must not be None!", op.opcode),
                    };
                    let inp0 = self
                        .varnode_read_value(&fn_ctx, &op.vars[0])
                        .into_int_value();
                    let t = inp0.get_type();
                    let t2 = self.int_type_from_length(outvar.size);
                    let _5 = t.const_int(0x5555555555555555, false);
                    let _3 = t.const_int(0x3333333333333333, false);
                    let _0f0 = t.const_int(0x0f0f0f0f0f0f0f0f, false);
                    let _00f = t.const_int(0x00ff00ff00ff00ff, false);
                    let _000 = t.const_int(0x0000ffff0000ffff, false);
                    macro_rules! pop_op {
                        ( $x:expr, $y:expr, $v:expr ) => {
                            self.builder.build_int_add(
                                self.builder.build_and($x, $y, "val"),
                                self.builder.build_and(
                                    self.builder.build_right_shift(
                                        $x,
                                        t.const_int($v, false),
                                        false,
                                        "val",
                                    ),
                                    $y,
                                    "val",
                                ),
                                "val",
                            )
                        };
                    }
                    let v0 = pop_op!(inp0, _5, 1);
                    let v1 = pop_op!(v0, _3, 2);
                    let v2 = pop_op!(v1, _0f0, 4);
                    let v3 = pop_op!(v2, _00f, 8);
                    let v4 = pop_op!(v3, _000, 16);
                    let r0 = self.builder.build_and(v4, t.const_int(0xff, false), "res0");
                    let r1 = self.builder.build_and(
                        self.builder
                            .build_right_shift(v4, t.const_int(32, false), false, "res1"),
                        t.const_int(0xff, false),
                        "res2",
                    );
                    let res = self.builder.build_int_cast(
                        self.builder.build_int_add(r0, r1, "res"),
                        t2,
                        "res",
                    );
                    self.varnode_write_value(
                        &mut fn_ctx,
                        outvar,
                        BasicValueEnum::IntValue(res),
                    );
                }
                _ => {
                    error!("No handler for opcode {:?}", op.opcode);
                    panic!("No handler for opcode {:?}", op.opcode);
                }
            }

            match op.opcode {
                PcodeOpCode::BRANCH | PcodeOpCode::CALL | PcodeOpCode::CBRANCH => {}
                _ => {
                    let next_bb = bb.get_next_basic_block();
                    if let Some(nbb) = next_bb {
                        self.builder.build_unconditional_branch(nbb);
                    }
                }
            }
            i += 1;
        }

        self.builder.position_at_end(ret_block);
        self.builder.build_return(None);
        trace!("{:?}", fn_ctx.func);
        unsafe { self.execution_engine.get_function(&fn_name).ok() }
    }
}

fn process_pcode(pcode_ops: &Vec<PcodeInstruction>) {
    let context = Context::create();
    let module = context.create_module("testing");
    let execution_engine = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    let codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };

    let mut emu_regs: [u8; 1024] = [0; 1024];
    let mut emu = Emu {
        regs: emu_regs.as_mut_ptr(),
    };
    let res_func = codegen.jit_compile_pcode(0, pcode_ops).unwrap();
    let start = Instant::now();
    for i in 0..10000 {
        emu_regs = [0; 1024];
        unsafe {
            res_func.call(&mut emu as *mut Emu);
        }
    }
    let elapsed = start.elapsed();
    debug!("regs: {:?}", emu_regs);
    debug!("Runtime: {:?}", elapsed);
    debug!("Time per run {}", elapsed.as_secs_f64() / 10000 as f64);
    debug!("Insts/sec {}", 10000 as f64 / elapsed.as_secs_f64());
}

fn read_a_file(fname: &str) -> std::io::Result<Vec<u8>> {
    let mut file = File::open(fname)?;

    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    return Ok(data);
}

fn main() {
    let matches = App::new("Pcode Fuzzing Engine")
        .version("1.0")
        .author("John W.")
        .about("Fuzzing with emulated pcodes")
        .arg(
            Arg::with_name("INPUT_BIN")
                .help("Input binary file")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name("v")
                .short("v")
                .multiple(true)
                .help("Sets level of verbosity for logging"),
        )
        .arg(
            Arg::with_name("LOG_FILE")
                .short("l")
                .long("logfile")
                .help("Sets a logfile to output to")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("ARCH")
                .short("a")
                .long("arch")
                .help("Sets architecture of the emulation")
                .takes_value(true),
        )
        .get_matches();
    let v_count = matches.occurrences_of("v");
    let verbosity = match v_count {
        0 => log::LevelFilter::Error,
        1 => log::LevelFilter::Warn,
        2 => log::LevelFilter::Info,
        3 => log::LevelFilter::Debug,
        4 | _ => log::LevelFilter::Trace,
    };

    if let Some(logfile_name) = matches.value_of("LOG_FILE") {
        CombinedLogger::init(vec![
            TermLogger::new(
                verbosity,
                Config::default(),
                TerminalMode::Mixed,
                ColorChoice::Auto,
            ),
            WriteLogger::new(
                log::LevelFilter::Trace,
                Config::default(),
                File::create(logfile_name).unwrap(),
            ),
        ])
        .unwrap();
    } else {
        TermLogger::init(
            verbosity,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        )
        .unwrap();
    }
    trace!("Logging initialized");

    let input_file_name = matches.value_of("INPUT_BIN").unwrap();
    info!("INPUT_BIN = \"{}\"", input_file_name);

    let arch_name = if let Some(a) = matches.value_of("ARCH") {
        a
    } else {
        warn!("Architecture not specified, defaulting to \"x86\"");
        "x86"
    };
    info!("ARCH = \"{}\"", arch_name);
    let spec = match arch(arch_name) {
        Ok(x) => x,
        Err(x) => {
            error!("Invalid architecture {}: \"{}\"", arch_name, x);
            panic!("Invalid architecture {}", arch_name);
        }
    };

    let filebuf = match read_a_file(input_file_name) {
        Ok(x) => x,
        Err(x) => {
            error!("Failed to read input file: \"{}\"", x);
            panic!("Failed to read input file");
        }
    };
    trace!("Input file: {:?}", filebuf);

    debug!("Beginning sleigh builder initialization");
    let mut sleigh_builder = SleighBuilder::default();
    let mut loader = PlainLoadImage::from_buf(&filebuf, 0);
    sleigh_builder.loader(&mut loader);
    sleigh_builder.spec(spec);
    sleigh_builder.mode(Mode::MODE64);

    debug!("Beginning disassembly");
    let mut asm_emit = CollectingAssemblyEmit::default();
    let mut pcode_emit = CollectingPcodeEmit::default();
    sleigh_builder.asm_emit(&mut asm_emit);
    sleigh_builder.pcode_emit(&mut pcode_emit);
    let mut sleigh = sleigh_builder.try_build().unwrap();

    debug!("Decoding...");
    sleigh.decode(0).unwrap();

    debug!("{:?}", asm_emit.asms);

    process_pcode(&pcode_emit.pcode_asms);
}
