
def error_response(message, details=None):
    resp = {"error": message}
    if details:
        resp["details"] = details
    return resp
