# scan.py

import json

def perform_face_scan():
    # TODO: Replace this with your actual face recognition logic
    result = {
        "status": "success",
        "message": "Face recognized successfully",
        "user": "John Doe"
    }
    return result

if __name__ == "__main__":
    result = perform_face_scan()
    print(json.dumps(result))  # Output result in JSON format
