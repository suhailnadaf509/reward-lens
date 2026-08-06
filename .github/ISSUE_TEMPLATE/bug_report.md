---
name: Bug report
about: Create a report to help us improve Reward Lens
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is. (e.g., "The `patch_activations` function raises a tensor mismatch error when using a custom reward head on ArmoRM.")

**To Reproduce**
Steps to reproduce the behavior, preferably with a minimal code example:
```python
from reward_lens import RewardModelAnalyzer
import torch

# Your code here
```

**Expected behavior**
What did you expect to happen? (e.g., "Expected the patched reward to match the scalar output size of the contrastive pair.")

**Environment Data:**
 - OS: [e.g. Ubuntu 22.04]
 - Python version [e.g. 3.11]
 - PyTorch version [e.g. 2.1.0]
 - Transformers variant [e.g. 4.35.0]
 - reward-lens version [e.g. 3.0.0]
 - Model being analyzed (if applicable) [e.g. Skywork/Skywork-Reward-Llama-3.1-8B]

**Additional Context or Output**
Add any other context about the problem here. Include full stack traces if the bug is an exception.
