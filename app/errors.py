class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def model_error(error: Exception) -> ServiceError:
    # Inspect the exception chain, but never expose/log Hub responses or tokens.
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(getattr(current, "response", None), "status_code", None)
        if status in (401, 403) or type(current).__name__ == "GatedRepoError":
            return ServiceError(503, "model_access_denied", "Check server HF_TOKEN and accepted model access terms.")
        current = current.__cause__ or current.__context__
    return ServiceError(503, "model_initialization_failed", "The speech model could not be initialized.")
