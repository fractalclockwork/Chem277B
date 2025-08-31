# kernel_registry.py

#############################
# lol, this got complicated
# we created this lbrary so that we fail gracefully if the required packages aren't present
# clearly we are not right in the head

import numpy as np


class KernelRegistry:
    """Registry for runtime-available palindrome kernels."""

    def __init__(self):
        """Register kernels based on available libraries."""
        self.kernels = {}

        try:
            from numba import njit

            self.register("numba", self._numba_kernel(njit))
        except ImportError:
            pass

        try:
            import torch

            self._torch = torch
            self.register("torch", self._torch_kernel())
        except ImportError:
            self._torch = None

        try:
            import jax
            import jax.numpy as jnp

            self.register("jax", self._jax_kernel(jnp))
        except ImportError:
            pass

        if self._torch and self._torch.cuda.is_available():
            try:
                import triton
                import triton.language as tl

                self.register("triton", self._triton_kernel(triton, tl))
            except ImportError:
                pass

    def register(self, name, func):
        """Add a kernel to the registry."""
        if func is not None:
            self.kernels[name] = func

    def get(self, name):
        """Retrieve a kernel by name."""
        return self.kernels.get(name)

    def available(self):
        """List registered kernel names."""
        return list(self.kernels.keys())

    def _numba_kernel(self, njit):
        """Numba-based kernel."""

        @njit
        def numba_palindrome(arr):
            left, right = 0, len(arr) - 1
            while left < right:
                if arr[left] != arr[right]:
                    return False
                left += 1
                right -= 1
            return True

        def wrapper(s):
            arr = np.frombuffer(s.encode("utf-8"), dtype=np.uint8)
            return numba_palindrome(arr)

        return wrapper

    def _torch_kernel(self):
        """PyTorch-based kernel."""

        def wrapper(s):
            device = self._torch.device(
                "cuda" if self._torch.cuda.is_available() else "cpu"
            )
            arr = np.frombuffer(s.encode("utf-8"), dtype=np.uint8)
            t = self._torch.from_numpy(arr).to(device)
            n = t.numel()
            if n <= 1:
                return True
            half = n // 2
            left = t[:half]
            right = self._torch.flip(t, dims=[0])[:half]
            return self._torch.equal(left, right)

        return wrapper

    def _jax_kernel(self, jnp):
        """JAX-based kernel."""

        def wrapper(s):
            arr = np.frombuffer(s.encode("utf-8"), dtype=np.uint8)
            x = jnp.array(arr)
            n = x.size
            if n <= 1:
                return True
            half = n // 2
            left = x[:half]
            right = jnp.flip(x)[:half]
            return bool(jnp.all(left == right))

        return wrapper

    def _triton_kernel(self, triton, tl):
        """Triton-based GPU kernel."""
        import torch

        @triton.jit
        def palindrome_kernel(x_ptr, n, out_ptr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            start = pid * BLOCK_SIZE
            offsets = start + tl.arange(0, BLOCK_SIZE)
            half = n // 2
            mask = offsets < half
            i = offsets
            j = (n - 1) - offsets
            a = tl.load(x_ptr + i, mask=mask, other=0)
            b = tl.load(x_ptr + j, mask=mask, other=0)
            neq = a != b
            mismatches = tl.sum(neq.to(tl.int32), axis=0)
            ok = (mismatches == 0).to(tl.int32)
            tl.store(out_ptr + pid, ok)

        def wrapper(s, block_size=4096):
            arr = np.frombuffer(s.encode("utf-8"), dtype=np.uint8)
            x = torch.from_numpy(arr).to("cuda", non_blocking=True)
            n = x.numel()
            if n <= 1:
                return True
            half = n // 2
            nprog = triton.cdiv(half, block_size)
            out = torch.empty(nprog, dtype=torch.int32, device="cuda")
            palindrome_kernel[(nprog,)](x, n, out, BLOCK_SIZE=block_size)
            return bool(torch.all(out == 1).item())

        return wrapper
