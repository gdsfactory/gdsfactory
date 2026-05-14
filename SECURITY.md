# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 9.x     | Latest minor only  |
| < 9.0   | :x:                |

Only the latest minor release of the current major version receives security updates.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to **security@gdsfactory.com** with the subject line: `[SECURITY] <brief description>`.

Include as much of the following as possible:

- Description of the vulnerability
- Steps to reproduce or a proof of concept
- Affected version(s)
- Potential impact

You should receive an initial response within **72 hours** acknowledging receipt. We will work with you to understand the issue and provide a timeline for a fix.

## Disclosure Policy

- We ask that you give us reasonable time to address the issue before any public disclosure.
- We will coordinate with you on the disclosure timeline and credit you (unless you prefer to remain anonymous).
- Once a fix is released, we will publish a security advisory on GitHub.

## Scope

This policy applies to the `gdsfactory` Python package and its official dependencies maintained under the [gdsfactory GitHub organization](https://github.com/gdsfactory).

## Security Best Practices for Users

- Pin your dependencies to specific versions in production environments.
- Keep gdsfactory updated to the latest supported version.
- Be cautious when loading GDS files or cell definitions from untrusted sources, as they may execute arbitrary Python code.
