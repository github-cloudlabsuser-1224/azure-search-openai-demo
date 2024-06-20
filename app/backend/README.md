# Backend Documentation

## Overview

The backend of this application is responsible for handling requests from the frontend, processing data, and interacting with Azure AI Search and other Azure resources.

## Getting Started

### Prerequisites

- Azure account
- Azure CLI installed
- Azure Developer CLI (azd) installed

### Deployment

If you've only changed the backend/frontend code in the [`app`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fazureuser%2Fazure-search-openai-demo%2Fapp%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\azureuser\azure-search-openai-demo\app") folder, you can redeploy the application with:

```sh
azd deploy
```

If you've changed the infrastructure files (`infra` folder or `azure.yaml`), you'll need to re-provision the Azure resources:

```sh
azd up
```

## Running Locally

You can only run locally **after** having successfully run the `azd up` command. 

1. Authenticate with Azure: `azd auth login`
2. Change directory to [`app`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fazureuser%2Fazure-search-openai-demo%2Fapp%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\azureuser\azure-search-openai-demo\app")
3. Start the project locally: `./start.ps1` or `./start.sh` or run the "VS Code Task: Start App"

## Hot Reloading

When you run `./start.ps1` or `./start.sh`, the backend files will be watched and reloaded automatically. 

## Sharing Environments

To give someone else access to a completely deployed and existing environment, follow these steps:

1. Install the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
2. Initialize the project: `azd init -t azure-search-openai-demo`
3. Refresh the environment: `azd env refresh -e {environment name}`
4. Set the `AZURE_PRINCIPAL_ID` environment variable to their Azure ID
5. Run [`./scripts/roles.ps1`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fazureuser%2Fazure-search-openai-demo%2Fscripts%2Froles.ps1%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\azureuser\azure-search-openai-demo\scripts\roles.ps1") or `.scripts/roles.sh` to assign all of the necessary roles to the user

## Additional Features

The backend supports a variety of optional features, including:

- GPT-4 with vision
- Speech input/output
- User login and data access via Microsoft Entra
- Performance tracing and monitoring with Application Insights

For more information on these features, refer to the [Additional Documentation](../../../../c:/Users/azureuser/azure-search-openai-demo/docs/README.md).

## Troubleshooting

If you encounter issues while working with the backend, refer to the [Debugging the app on App Service](../../../../c:/Users/azureuser/azure-search-openai-demo/docs/appservice.md) guide.

## Contributing

Contributions to the backend are welcome. Please ensure that you follow the established coding standards and that all tests pass before submitting a pull request.

## License

This project is licensed under the terms of the [insert license here] license.
