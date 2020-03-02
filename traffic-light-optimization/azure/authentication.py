
from msrestazure.azure_active_directory import AADTokenCredentials

import adal, uuid, time


def authenticate_client():
    """
    Authenticate using service principal w/ key.
    """
    authority_host_uri = 'https://login.microsoftonline.com'
    tenant = '960eee66-f2c8-4a21-917c-33ae2876ae0c' #'<TENANT>'
    authority_uri = authority_host_uri + '/' + tenant
    resource_uri = 'https://management.core.windows.net/'
    client_id = '842bc46f-15d5-41d2-a4a7-990b02d32b61' #'<CLIENT_ID>'
    client_secret = 'EgNi8kcfDZwWcbSWARHwJYm3XjMtIilG+xJb7DK0s38=' #'<CLIENT_SECRET>'

    context = adal.AuthenticationContext(authority_uri, api_version=None)
    management_token = context.acquire_token_with_client_credentials(resource_uri, client_id, client_secret)
    credentials = AADTokenCredentials(management_token, client_id)

    return credentials