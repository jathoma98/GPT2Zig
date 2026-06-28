IMPORTANT: Keep your writings in this file concise and to the point to avoid context bloat.

## Project
This is a simple fully-CPU implemented Zig inference engine for GPT2.
The user is an LLM n00b who is using this to learn LLM inference on bare metal.

## Claude Response Style
- Prefer a direct, concise communication style to save tokens and thinking time. 
The user wants a low latency, low friction workflow -- skip unnecessary flattery,
focus on delivery and value.
- The user is a Zig/C++/Systems Programming expert but a total n00b to LLMs.
Therefore, whenever the user asks anything LLM related, think deeply about the user is trying to accomplish.
If it sounds wrong to you (Claude), it probably is, and offer clarification to the user.
The user won't feel patronized: in fact, the user is here to learn about LLMs and will be
delighted to learn something new from you.

## Zig Coding Style Guidelines

- For 'code smell' guidelines, remember that 'code smell' does not equal 'banned' -- it means the code has
typically unfavorable characteristics. You are welcome to use any 'code smell' so long as you provide a proper
justification (appeal to simplicity, perf, etc.)

- Prefer simplicity and explicitness. Introduce abstractions only when they have a justifiable complexity vs simplicity tradeoff.

- Move errors as close to their source as possible. Compile time is ideal.
 When that isn't possible, assert program invariants at runtime (preconditions and postconditions), as close to the point
 of invariant violation as possible. Runtime asserts should be performed especially
 aggressively during init code where the perf price of such invariants is low relative to
 whole program execution.

 - `zig build test` should test everything. If a module with module-level tests isn't discoverable from the root module,
 use the:
 ```
 test {
    _ = @import("module);
 }
 ```
 idiom to force discovery from the main module.

- Prefer a functional, explicit style of programming. Tagged unions are excellent for encoding state machines
and state private data: Zig enforces via `switch` that a state's private data can only be accessed when the
variable is actually in that state.

- Prefer explicit initialization of struct members over `undefined` or zero-initting then setting members
manually line-by-line. Constructing a struct with the `.{ member = value, }` syntax ensures we get compile errors
when we add new members. Default values for members are okay so long as they are actually valid default values
and not simply means of silencing the compiler to be overwritten at runtime later. The `.init(args)` idiom
is ideal for structs with nontrivial initialization.

- Explanatory comments are a code smell. Comments should explain 'why', not 'what'. They should explain unintuitive tradeoffs,
or surprising runtime behavior, etc. 
Exception: Structural comments in large code blocks so that the large function can be grokked at a glance. These are encouraged.
`parseHeader` in `safetensors.zig` is a good example of structural commenting.

- Helper functions are a code smell - good justifications are: multiple usage sites for a common codepath,
testability (ex: a state transition function so that it can be tested). For organizing large code blocks, prefer
header-style comments like:
```
// =================
// === Section A ===
```

- Heap allocations are a code smell - prefer to reason about a likely upper bound for memory needed and statically
allocate within that upper bound. Of course, sometimes heap allocation is unavoidable -- just make sure you provide
a proper justification if you reach for it.

- Raw pointers are a code smell - prefer opaque enum handle types (const Handle = enum(u32) {_} ), which are array indices
into some flat array of data structures. They are often more compact (not needing a full 64 bit address) and can optionally
embed safety semantics.

- For stateful code, you should reach for the 'reduce -> decide -> transition pattern'. A 'reduce' function is an ideally
pure function which takes disparate program state data and reduces it to a 'decision' output, usually a tagged union specifying
a decision based on the input state. A 'decider' is a switch over the 'reduce' function's output -- it determines what should
be done based on the 'reduce' decision. 'transition' functions take a particular state machine and transitions it within
a 'decider' switch block.
Ex:
```
1. User clicks mouse -> produces MouseInputData
2. reducer(MouseInputData, ProgramState) -> OpenDialogIntent (based on program state and user input, we reduce to a concrete user intent)
3. switch (OpenDialogIntent) -> case OpenFoobarDialog |fb| { transitionFoobarState(fb) } (switch on decision, transition the relevant state)
```
This allows complex state to read as a straight line algorithm that is trivially unit testable.

- Prefer transitioning state in multiple 'passes', where after each pass some program invariant can be assumed to be fully decided upon.
Ex: imagine we have some program which consumes input and polls background jobs. We would encode this as multiple passes:
```
// =================
// === Resolve Input ===

// ... some code that takes raw input and reduces to a decision

// =================
// === Poll State ===

// ... this code can assume the input decision for this tick is final
// since it was already calculated in a previous pass.

```
This allows one to 'code with confidence', as opposed to having to constantly defensively check for nulls or unset state.
Additionally, it creates a single source of truth for state, which aids code organization.

- 'null' is a code smell. 'null' does a poor job of defining explicit intent: prefer to use tagged union state machines.
Acceptable 'null' (or optional '?T' type) usage is as the return of a function which can genuinely return a null state
that the user is expected to switch on and handle, or for cases where the 'null' encodes semantics that are obvious
at first glance (ex: 'null' in a json field means nonexistent).

- Prefer functional style composition over object-oriented style member functions. Structs are containers for data,
functions perform transformations on data. Keep functions pure if the perf and complexity characteristics are favorable.
Good exceptions for member functions are container types like Lists where List.push() makes sense semantically as the list 'owning' its accounting
data structures. Good exceptions for non-pure functions are top-level functions with large scopes, ex: a frame tick in a video game, where
copying every tick's input and output data would have prohibitive perf characteristics.

- Avoid magic numbers: encode constants as enums for readability and correctness.


## Python reference oracle

All Python tooling lives under `python/` with a venv at `python/.venv`.
