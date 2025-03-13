import os
import time
from enum import Enum
from typing import NamedTuple, Optional

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import ClientSecretCredential

import mlflow

REQUEST_URL_CHAT = "https://api.openai.com/v1/chat/completions"
REQUEST_URL_COMPLETIONS = "https://api.openai.com/v1/completions"
REQUEST_URL_EMBEDDINGS = "https://api.openai.com/v1/embeddings"

REQUEST_FIELDS_CHAT = {
    "model",
    "messages",
    "frequency_penalty",
    "logit_bias",
    "max_tokens",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "user",
    "function_call",
    "functions",
}
REQUEST_FIELDS_COMPLETIONS = {
    "model",
    "prompt",
    "best_of",
    "echo",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_tokens",
    "n",
    "presence_penalty",
    "seed",
    "stop",
    "stream",
    "suffix",
    "temperature",
    "top_p",
    "user",
}
REQUEST_FIELDS_EMBEDDINGS = {"input", "model", "encoding_format", "user"}
REQUEST_FIELDS = REQUEST_FIELDS_CHAT | REQUEST_FIELDS_COMPLETIONS | REQUEST_FIELDS_EMBEDDINGS


def _validate_model_params(task, model, params):
    if not params:
        return

    if any(key in model for key in params):
        raise mlflow.MlflowException.invalid_parameter_value(
            f"Providing any of {list(model.keys())} as parameters in the signature is not "
            "allowed because they were indicated as part of the OpenAI model. Either remove "
            "the argument when logging the model or remove the parameter from the signature.",
        )
    if "batch_size" in params and task == "chat.completions":
        raise mlflow.MlflowException.invalid_parameter_value(
            "Parameter `batch_size` is not supported for task `chat.completions`"
        )


class _OAITokenHolder:
    def __init__(self, api_type):
        self._credential = None
        self._api_type = api_type
        self._is_azure_ad = api_type in ("azure_ad", "azuread")
        self._azure_ad_token = None
        self._api_token_env = os.environ.get("OPENAI_API_KEY")
        self.entra_scope = os.environ.get("AZURE_ENTRA_SCOPE")

        if self._is_azure_ad and not self._api_token_env:
            try:
                tenant_id = os.getenv("AZURE_TENANT_ID")
                client_id = os.getenv("AZURE_CLIENT_ID")
                client_secret = os.getenv("AZURE_CLIENT_SECRET")

                if not all([tenant_id, client_id, client_secret]):
                    raise mlflow.MlflowException(
                        "Using API type `azure_ad` or `azuread` requires the environment "
                        "variables `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and `AZURE_CLIENT_SECRET` "
                        "to be set."
                    )

                self._credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret,
                )
            except ImportError:
                raise mlflow.MlflowException(
                    "Using API type `azure_ad` or `azuread` requires the package "
                    "`azure-identity` to be installed."
                )

    @property
    def token(self):
        return self._api_token_env or self._azure_ad_token.token

    def refresh(self, logger=None):
        """Validates the token or API key configured for accessing the OpenAI resource."""

        if self._api_token_env is not None:
            return

        if self._is_azure_ad:
            if not self._azure_ad_token or self._azure_ad_token.expires_on < time.time() + 60:
                if logger:
                    logger.debug(
                        "Token for Azure AD is either expired or unset. Attempting to "
                        "acquire a new token."
                    )
                try:
                    self._azure_ad_token = self._credential.get_token(self.entra_scope)
                except ClientAuthenticationError as err:
                    raise mlflow.MlflowException(
                        "Unable to acquire a valid Azure AD token for the resource due to "
                        f"the following error: {err.message}"
                    ) from err

            if logger:
                logger.debug("Token refreshed successfully")
        else:
            raise mlflow.MlflowException(
                "OpenAI API key must be set in the ``OPENAI_API_KEY`` environment variable."
            )


class _OpenAIApiConfig(NamedTuple):
    api_type: str
    batch_size: int
    max_requests_per_minute: int
    max_tokens_per_minute: int
    api_version: Optional[str]
    api_base: str
    deployment_id: Optional[str]
    organization: Optional[str] = None
    max_retries: int = 5
    timeout: float = 60.0
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    entra_scope: Optional[str] = None  # NEW FIELD


# See https://github.com/openai/openai-python/blob/cf03fe16a92cd01f2a8867537399c12e183ba58e/openai/__init__.py#L30-L38
# for the list of environment variables that openai-python uses
class _OpenAIEnvVar(str, Enum):
    OPENAI_API_TYPE = "OPENAI_API_TYPE"
    OPENAI_BASE_URL = "OPENAI_BASE_URL"
    OPENAI_API_BASE = "OPENAI_API_BASE"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    OPENAI_API_KEY_PATH = "OPENAI_API_KEY_PATH"
    OPENAI_API_VERSION = "OPENAI_API_VERSION"
    OPENAI_ORGANIZATION = "OPENAI_ORGANIZATION"
    OPENAI_ENGINE = "OPENAI_ENGINE"
    OPENAI_DEPLOYMENT_NAME = "OPENAI_DEPLOYMENT_NAME"

    # Azure Credentials for authentication
    AZURE_TENANT_ID = "AZURE_TENANT_ID"
    AZURE_CLIENT_ID = "AZURE_CLIENT_ID"
    AZURE_CLIENT_SECRET = "AZURE_CLIENT_SECRET"
    AZURE_ENTRA_SCOPE = "AZURE_ENTRA_SCOPE"

    @property
    def secret_key(self):
        return self.value.lower()

    @classmethod
    def read_environ(cls):
        env_vars = {}
        for e in _OpenAIEnvVar:
            if value := os.getenv(e.value):
                env_vars[e.value] = value
        return env_vars
