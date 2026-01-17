# Contributing to Chassis

Thank you for looking at this project. Chassis is built to be a simple and stable foundation for others to use, and it only stays that way through careful work. Whether you are fixing a bug, improving the documentation, or helping someone else with a question, your help is appreciated.

If you like what we are doing but do not have the time to write code or documentation, you can still help by starring the repository or mentioning the project to others who might find a local vector engine useful.

## I Have a Question

We try to keep the [documentation](https://tanvincible.github.io/chassis/) clear and up to date in the project book. Before asking a question, please check the book to see if the answer is already there. If it is not, search the existing [issues](https://github.com/tanvincible/chassis/issues) on GitHub to see if someone else has already asked the same thing.

If you still need help, feel free to [open an issue](https://github.com/tanvincible/chassis/issues/new
). Please be as specific as possible about what you are trying to do and what is not working. Mentioning your operating system and the version of Chassis you are using helps a lot.

## How to Contribute

We value stability and correctness above all else. By contributing, you agree that your work is your own and that it may be distributed under the project’s MIT and Apache 2.0 licenses.

### Reporting Bugs

A good bug report helps us fix things quickly without needing to ask for more information later.

Before reporting a bug, make sure you are using the latest version and check the documentation once more to ensure the behavior is not intended. Please also search the [issue tracker](https://github.com/tanvincible/chassis/issues?q=label%3Abug) to see if the bug has already been reported.

When reporting a bug, include:

1. A clear description of what happened and what you expected to happen
2. The steps needed to reproduce the issue. A small, isolated code example is ideal
3. Your environment details such as operating system (Linux, Windows, macOS) and architecture (x86, ARM)
4. Any error messages or stack traces

### Security Issues

If you discover a security vulnerability, please do not report it publicly. Instead, send an email to **[tanvipm19@gmail.com](mailto:tanvipm19@gmail.com)** so it can be handled responsibly.

### Suggesting Enhancements

Chassis is intentionally small and focused. If you have an idea for a new feature, please consider whether it fits the project’s goal of being a simple storage primitive.

We prefer features that benefit most users rather than specialized functionality that can be built on top of Chassis by the host application.

When suggesting an enhancement, explain why it would be useful and how you imagine it working. If other tools solve a similar problem well, you may reference them as examples.

### Your First Code Contribution

To start working on the code, you will need a recent stable version of Rust. The project uses a workspace layout, with most core logic living in the `chassis-core` directory.

1. Fork the repository and clone it to your machine
2. Create a new branch for your work
3. Ensure your changes include tests that verify the fix or feature
4. Run `cargo fmt` to keep the code style consistent
5. Run `cargo clippy` to catch common issues
6. Run `cargo test` to ensure all tests pass

When your code is ready, open a pull request. All contributions are reviewed with a focus on correctness, maintainability, and performance.

### Improving Documentation

Documentation is as important as code. We use mdbook to manage the documentation, which lives in the [`docs`](https://github.com/tanvincible/chassis/tree/main/docs
) directory.

If you find a typo, unclear explanation, or missing detail, please feel free to submit a pull request with the correction.

## Style Guidelines

### Commit Messages

We use a simple commit message format to keep the project history readable and useful over time.

Each commit message should start with a prefix that describes the type of change:

- feat: A new feature
- fix: A bug fix
- docs: Documentation changes
- build: Build system or dependency changes
- refactor: Code changes that do not alter behavior
- perf: Performance improvements
- test: Adding or fixing tests
- chore: Maintenance tasks and tooling changes

A typical commit message looks like:

`feat: add support for cosine similarity in the core engine`

For larger changes, you may include a message body after a blank line with additional context. A git commit template is provided in the [repository](https://github.com/tanvincible/chassis/blob/main/.gitmessage).

## Final Note

This project exists because we believe in local, reliable software. We are glad you are here to help build it.
