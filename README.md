# PCFE

**PC**ode **F**uzzing **E**ngine

PCFE is an attempt at creating a JIT'd Pcode emulation engine for fuzzing purposes.

Currently we just JIT the provided binary file into a function and then call it 10000x to determine how fast we are.

## Planned features

* Branch coverage
* Compare coverage
* Tie into [LibAFL](https://github.com/AFLplusplus/LibAFL)
* Provide an interface for using with libfuzzer/AFL/hongfuzz

## Requirements

* Rust 1.53
* llvm 10.0+

## Usage

```
cargo run <target bin> [-vvvv] [-a ARCH]
```

