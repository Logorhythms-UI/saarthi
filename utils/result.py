from typing import Any
# This Module Consists of Helper Utilities


# This is a result class.
# I really like the way Rust and Go do error handling.
# I prefer that kind of error handling over try catch blocks.
# Hence this Result class that behaves like the Result<T,E> in rust.
class Result:
    def __init__(self, result: Any = None, err: str | None = None) -> None:
        assert isinstance(err, str | None), "Error can only be a string or None"

        self.result = result  # Actual Result
        self.err = err  # Error, if any
        self.ok = err is None  # Ok says if result is available or not
