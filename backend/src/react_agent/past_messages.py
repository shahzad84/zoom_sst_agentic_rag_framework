# def transform_past_messages(data):
#     output = []
#     for entry in data:
#         for key in ['user', 'bot']:
#             if key in entry:
#                 output.append({
#                     "role": "assistant" if key == "bot" else "user",
#                     "content": entry[key]["text"]
#                 })
#     return output