# def calculator(expression):
#     return eval(expression)

def calculator(expression):
    if expression == "FAIL":
        raise ValueError("Calculator failed intentionally")
    
    return eval(expression)

def get_weather(city):
    return f"The weather in {city} is sunny and 72°F."


def broken_tool():
    raise Exception("Something went wrong with this tool")