import boto3
import json
from datetime import date
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

session = boto3.Session()
region = session.region_name

modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'

bedrock_client = boto3.client(service_name = 'bedrock-runtime', region_name = region,)

def search_duckduckgo(query, region='wt-wt', safesearch='off', max_results=5):
    """Search DuckDuckGo (ddg) for the given query and return the results. This is for websearch, we need this for current information."""
    ddg = DDGS()
    results = ddg.text(keywords=query, region=region, safesearch=safesearch, max_results=max_results)
    return results

def scrape_webpage(url):
    """Scrape a webpage and return its content as text."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)
    return text

def process_tool_call(tool_name, tool_input):
    if tool_name == "search":
        return search_duckduckgo(tool_input["query"])
    elif tool_name == "webscrape":
        return scrape_webpage(tool_input["url"])

toolConfig = {
    'tools': [
        {
            'toolSpec': {
                'name': 'search',
                'description': 'This tool allows you to search the web using DuckDuckGo. You can use it to find information, articles, websites, and more. Simply provide a query, and the tool will return a list of search results.',
                'inputSchema': {
                    'json': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The search query. This can be any string of text that you want to search for.'
                            }
                        },
                        'required': ['query']
                    }
                }
            }
        },
        {
            'toolSpec': {
                'name': 'webscrape',
                'description': 'This tool allows you to scrape the content of a webpage. You can use it to extract the text from a webpage, which can then be used as context for further actions. Simply provide a URL, and the tool will return the text content of the webpage.',
                'inputSchema': {
                    'json': {
                        'type': 'object',
                        'properties': {
                            'url': {
                                'type': 'string',
                                'description': 'The URL of the webpage to scrape. This should be a fully qualified URL, including the http:// or https:// prefix.'
                            }
                        },
                        'required': ['url']
                    }
                }
            }
        }
    ],
    'toolChoice': {
        'auto': {}
    }
}

def simple_chat():
    user_message = input("\nUser: ")
    messages = [{"role": "user", "content": [{"text": user_message}]}]
    system_prompt = f"""
    Answer as many questions as you can using your existing knowledge.
    Only search the web for queries that you can not confidently answer.
    Today's date is {date.today().strftime("%B %d %Y")}
    If you think a user's question involves something in the future that hasn't happened yet, use the search tool.
    """
    while True:
        if messages[-1].get("role") == "assistant":
            user_message = input("\nUser: ")
            messages.append({"role": "user", "content": [{"text": user_message}]})

        converse_api_params = {
            "modelId": modelId,
            "system": [{"text": system_prompt}],
            "messages": messages,
            "inferenceConfig": {"temperature": 0.0, "maxTokens": 1000},
            "toolConfig":toolConfig,
        }

        response = bedrock_client.converse(**converse_api_params)

        messages.append({"role": "assistant", "content": response['output']['message']['content']})

        if response['stopReason'] == "tool_use":
            tool_use = response['output']['message']['content'][-1]
            tool_id = tool_use['toolUse']['toolUseId']
            tool_name = tool_use['toolUse']['name']
            tool_input = tool_use['toolUse']['input']

            print(f"Claude wants to use the {tool_name} tool")
            print(f"Tool Input:")
            print(json.dumps(tool_input, indent=2))

            tool_result = process_tool_call(tool_name, tool_input)

            print(f"\nTool Result:")
            print(json.dumps(tool_result, indent=2))

            messages.append({
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_id,
                            "content": [
                                {"text": str(tool_result)}
                            ]
                        }
                    }
                ]
            })

        else: 
            print("\nTechNova Support:" + f"{response['output']['message']['content'][0]['text']}")

simple_chat()