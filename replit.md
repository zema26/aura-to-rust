# Aura Compiler

A compiler that translates Aura programming language source code into Rust source code.

## Overview

The Aura language features unique syntax:
- **Reversed assignments**: `expr -> variable` (assigns expr to variable)
- **Tagged keywords**: `{while}...{/while}`, `{fun}...{/fun}`
- **Pipeline expressions**: `in -> a, b -> Function -> out`
- **Variable initialization**: `int a(0)` declares and initializes
- **For loops**: `{for} i(0)++ < n` means i from 0 to n-1

## Project Structure

- `src/main.rs` - Main compiler implementation (lexer, parser, code generator)
- `Aura1.txt` - Example: Euclidean algorithm
- `Aura2.txt` - Example: Eratosthenes sieve
- `example.aura` - Simple demonstration program

## Usage

```bash
cargo run -- <aura_file.aura>
```

This will output Rust source code to stdout.

## Architecture

1. **Lexer**: Tokenizes Aura source code, handling tagged keywords `{keyword}`
2. **Parser**: Builds AST with support for:
   - Module and function declarations
   - Control flow (if/else, while, for)
   - Variable declarations with initialization
   - Pipeline expressions for input/output
3. **Code Generator**: Translates AST to Rust code

## Example

Input (Aura):
```aura
{module} Example
  {fun} int a, int b -> add int
    {return} a + b {/return}
  {/fun}
{/module}
```

Output (Rust):
```rust
fn add(a: i32, b: i32) -> i32 {
    return (a + b);
}
```

## Known Limitations

- Pipeline expression parsing is simplified for common patterns
- Complex multi-step pipelines may not generate optimal code
- Some edge cases in expression parsing need refinement

## Workflow

The "Aura Compiler Demo" workflow compiles example.aura to demonstrate the compiler in action.
