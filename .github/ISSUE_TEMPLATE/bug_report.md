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

- [ ] I have reviewed the documentation, discussions and issues and found no relevant solution.
- [ ] I understand that this is an open-source project, and maintainers may not have the resources to address every issue. I am prepared to contribute by fixing the issue myself or hiring someone to do so, if needed, and accept that the issue may not be resolved otherwise.
- [ ] I am using the latest version of GDSFactory, with Python 3.10, 3.11, or 3.12.

Please provide the output for the following code:


```python
import sys
print(sys.version)
print(sys.executable)

import gdsfactory as gf
gf.config.print_version_plugins()
```
