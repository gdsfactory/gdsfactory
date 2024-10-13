---
name: Bug report
about: Create a report to help us improve
title: ''
labels: 'bug'
assignees: ''

---

**Describe the bug**
What's the bug?

**To Reproduce**
What's the code to reproduce the behavior? What commands or code did you write to get the error?
Make sure you include the all code needed for others to reproduce your issue.

**Expected behavior**
What would you like to happen?

**Suggested fix**
How could we fix the bug? As an open-source project, we welcome your suggestions and PRs.


**Environment (please complete the following information):**

- [ ] I have reviewed the documentation but found no relevant solution.
- [ ] I have searched the existing issues and found no resolution.
- [ ] I acknowledge that, as an open-source project, the maintainers may not have the capacity to address this issue. I am willing to contribute by fixing it myself or hiring someone to resolve it (contact@gdsfactory.com). Otherwise, I understand that the issue may remain unresolved.
- [ ] I am using the latest version of GDSFactory with Python 3.11, or 3.12. Below is the output for the following code

```python
import sys
print(sys.version)
print(sys.executable)

import gdsfactory as gf
gf.config.print_version_plugins()
```
