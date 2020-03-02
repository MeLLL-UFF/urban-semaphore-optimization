
import azure.authentication as AzureAuthentication

credentials = AzureAuthentication.authenticate_client()

print(credentials)