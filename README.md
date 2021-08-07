# PCFE

**PC**ode **F**uzzing **E**ngine

PCFE is an attempt at creating a JIT'd Pcode emulation engine for fuzzing purposes.

Currently just JIT some code from the provided binary file, call it, and display some register information.

## Todo List

* Add big endian support
* Fix the weird bug with branching + optimizations
* Breakpoints
* Coverage callbacks
* Emulator API
  * Python bindings for ease of use
  * Loading snapshots
    * Also may require Sleighcraft modifications

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

