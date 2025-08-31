import numpy as np
from kernel_registry import KernelRegistry


class Palindrome_Checker:
    """
    A utility class for checking whether a string is a palindrome using various methods.

    Parameters
    ----------
    input_str : str
        The string to be tested.

     mode : str, optional
        The method used for palindrome checking. Options include:
        'loop', 'break_loop', 'pythonic', 'map', 'memoryview', 'numba',
        'numpy', 'recursive', 'torch', 'triton', 'jax'.
        Default is 'loop'.
    """

    def __init__(self, input_str, mode="loop"):
        """Initialize with input string and test mode."""
        assert isinstance(
            input_str, str
        ), f"Expected a string, got {type(input_str).__name__}"
        assert isinstance(mode, str), f"Expected a string, got {type(mode).__name__}"
        self.input_str = input_str
        self.mode = mode
        self.input_len = len(input_str)
        self.registry = KernelRegistry()

        self.dispatch = {
            "loop": self.loop_test,
            "break_loop": self.break_loop_test,
            "pythonic": self.pythonic_test,
            "map": self.map_test,
            "memoryview": self.memoryview_test,
            "numpy": self.numpy_test,
            "recursive": lambda: self.recursive_test(self.input_str),
        }

        # Add our dynamic kernels to dispatch
        for name in self.registry.available():
            self.dispatch[name] = lambda name=name: self.registry.get(name)(
                self.input_str
            )

    def available(self):
        """Return all supported modes as a list."""
        return list(self.dispatch.keys())

    def run_mode(self, mode):
        """Set mode and return a statement about palindrome status."""
        self.mode = mode
        result = self.is_palindrome()
        return f"Mode '{mode}': {'Palindrome' if result else 'Not a palindrome'}"

    def is_palindrome(self):
        """Dispatch to selected method."""
        if self.input_len == 0:
            return True

        try:
            return self.dispatch[self.mode]()
        except KeyError:
            raise ValueError(
                f"Invalid mode: {self.mode}. Available: {self.available()}"
            )

    def pythonic_test(self) -> bool:
        """Check palindrome using string reversal."""
        return self.input_str == self.input_str[::-1]

    def map_test(self) -> bool:
        """Check palindrome using map and zip for character comparison."""
        half = self.input_len // 2
        left = self.input_str[:half]
        right = self.input_str[-1 : -half - 1 : -1]
        return all(map(lambda x: x[0] == x[1], zip(left, right)))

    def loop_test(self) -> bool:
        """Check palindrome using direct indexing and early return."""
        for i in range(self.input_len // 2):
            if self.input_str[i] != self.input_str[-i - 1]:
                return False
        return True

    def break_loop_test(self) -> bool:
        """Check palindrome using loop and break statement."""
        is_palindrome = True
        for i in range(self.input_len // 2):
            if self.input_str[i] != self.input_str[-i - 1]:
                is_palindrome = False
                break
        return is_palindrome

    def recursive_test(self, sub_str, left=0, right=None) -> bool:
        """Check palindrome recursively using index comparison."""
        if right is None:
            right = len(sub_str) - 1

        if left >= right:
            return True
        if sub_str[left] != sub_str[right]:
            return False
        return self.recursive_test(sub_str, left + 1, right - 1)

    def memoryview_test(self) -> bool:
        """Check palindrome using memoryview for byte-level comparison."""
        b = memoryview(self.input_str.encode("utf-8"))
        left, right = 0, len(b) - 1
        while left < right:
            if b[left] != b[right]:
                return False
            left += 1
            right -= 1
        return True

    def numpy_test(self) -> bool:
        """Check palindrome using NumPy array and slicing."""
        arr = np.frombuffer(self.input_str.encode("utf-8"), dtype=np.uint8)
        return np.array_equal(arr, arr[::-1])

    def pythonic_test(self) -> bool:
        """Check palindrome using string reversal."""
        return self.input_str == self.input_str[::-1]

    def map_test(self) -> bool:
        """Check palindrome using map and zip for character comparison."""
        half = self.input_len // 2
        left = self.input_str[:half]
        right = self.input_str[-1 : -half - 1 : -1]
        return all(map(lambda x: x[0] == x[1], zip(left, right)))

    def loop_test(self) -> bool:
        """Check palindrome using direct indexing and early return."""
        for i in range(self.input_len // 2):
            if self.input_str[i] != self.input_str[-i - 1]:
                return False
        return True

    def break_loop_test(self) -> bool:
        """Check palindrome using loop and break statement."""
        is_palindrome = True
        for i in range(self.input_len // 2):
            if self.input_str[i] != self.input_str[-i - 1]:
                is_palindrome = False
                break
        return is_palindrome

    def recursive_test(self, sub_str, left=0, right=None) -> bool:
        """Check palindrome recursively using index comparison."""
        if right is None:
            right = len(sub_str) - 1

        if left >= right:
            return True
        if sub_str[left] != sub_str[right]:
            return False
        return self.recursive_test(sub_str, left + 1, right - 1)

    def memoryview_test(self) -> bool:
        """Check palindrome using memoryview for byte-level comparison."""
        b = memoryview(self.input_str.encode("utf-8"))
        left, right = 0, len(b) - 1
        while left < right:
            if b[left] != b[right]:
                return False
            left += 1
            right -= 1
        return True

    def numpy_test(self) -> bool:
        """Check palindrome using NumPy array and slicing."""
        arr = np.frombuffer(self.input_str.encode("utf-8"), dtype=np.uint8)
        return np.array_equal(arr, arr[::-1])
