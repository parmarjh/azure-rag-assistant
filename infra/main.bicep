param location string = resourceGroup().location
param appName string = 'northwind-rag-assistant'

resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: '${appName}-search'
  location: location
  sku: { name: 'basic' }
  properties: { replicaCount: 1, partitionCount: 1, hostingMode: 'default' }
}

resource openai 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: '${appName}-openai'
  location: location
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: { customSubDomainName: '${appName}-openai' }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${appName}-insights'
  location: location
  kind: 'web'
  properties: { Application_Type: 'web' }
}

resource web 'Microsoft.Web/sites@2022-03-01' = {
  name: appName
  location: location
  kind: 'app,linux'
  properties: { siteConfig: { linuxFxVersion: 'PYTHON|3.10' } }
}
