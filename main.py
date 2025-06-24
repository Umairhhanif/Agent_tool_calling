from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv
import requests
import random
import chainlit as cl


load_dotenv()
set_tracing_disabled(disabled=True)



gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(

    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"

)

model=OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)


@function_tool
def how_many_jokes():
    """
    Get Random Number for Jokes 
    """
    return random.randint(1, 10)

@function_tool
def get_weather(city: str) -> str:
    """
    Get the weather for a given city
    """
    try:
        result = requests.get(
            f"https://api.weatherapi.com/v1/current.json?key=8e3aca2b91dc4342a1162608252604&q={city}"
        )
        data = result.json()
        return f"The current weather in {city} is {data['current']['temp_c']}C with {data['current']['condition']['text']}"
    except Exception as e:
        return f"Error getting weather: {str(e)}"


@cl.on_chat_start
async def start():
    # Initialize agent
    agent = Agent(
        name="Assistant",
        instructions="""
        if the user asks for jokes, use the how_many_jokes tool to get a random number of jokes.
        if the user asks for the weather, use the get_weather tool to get the weather for a given city.
        """,
        model=model,
        tools=[get_weather, how_many_jokes],
    )
    
    # Initialize message history
    cl.user_session.set("message_history", [])
    
    # Store agent in user session
    cl.user_session.set("agent", agent)

@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    message_history = cl.user_session.get("message_history")
    
    # Add user message to history
    message_history.append({"role": "user", "content": message.content})
    
    # Create a message with a thinking indicator
    thinking_msg = cl.Message(content="", author="Assistant")
    await thinking_msg.send()
    
    # Run the agent
    result = await cl.make_async(Runner.run_sync)(
        agent,
        input=message.content,
    )
    
    # Update the message with the final output
    final_output = result.final_output
    thinking_msg.content = final_output  # Set content property first
    await thinking_msg.update()  # Then update without parameters
    
    # Add assistant's response to history
    message_history.append({"role": "assistant", "content": final_output})
    
    # Update session history
    cl.user_session.set("message_history", message_history)

# This section is for running directly, not needed when running with 'chainlit run'
if __name__ == "__main__":
    # Comment out or remove this section when running with 'chainlit run main.py'
    agent = Agent(
        name="Assistant",
        instructions="""
        if the user ask for jokes, use the how_many_jokes tool to get a random number of jokes.
        if the user ask for the weather, use the get_weather tool to get the weather for a given city.
        """,
        model=model,
        tools=[get_weather, how_many_jokes],
    )

    result = Runner.run_sync(
        agent,
        input="tell me karachi weather",
    )

    print(result.final_output)

