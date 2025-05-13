import cv2
import os
import numpy as np

def detect_faces(image_path, cascade_path='haarcascade_frontalface_default.xml'):
    """
    Detects faces in an image using a Haar cascade classifier.

    Args:
        image_path (str): Path to the input image.
        cascade_path (str): Path to the Haar cascade XML file.

    Returns:
        list: A list of tuples, where each tuple contains (x, y, w, h) for a detected face.
              Returns an empty list if no faces are detected or the image cannot be read.
              Returns None if the cascade classifier cannot be loaded.
    """
    if not os.path.exists(cascade_path):
        # Fallback to a common system path if the local one is not found
        # This might need adjustment based on the OpenCV installation
        cascade_path_system = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path_system):
            cascade_path = cascade_path_system
        else:
            print(f"Error: Haar cascade file not found at {cascade_path} or system default path.")
            # As a last resort, try to download it if we had internet access and rights
            # For now, we will return None to indicate critical failure.
            # In a real application, you might attempt to download it here.
            # For example:
            # import urllib.request
            # url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            # print(f"Attempting to download Haar cascade from {url}...")
            # try:
            #     urllib.request.urlretrieve(url, 'haarcascade_frontalface_default.xml')
            #     cascade_path = 'haarcascade_frontalface_default.xml'
            #     print("Download successful.")
            # except Exception as e:
            #     print(f"Failed to download Haar cascade: {e}")
            #     return None # Critical error, cannot proceed
            print("Please ensure 'haarcascade_frontalface_default.xml' is available.")
            return None


    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Error: Could not load Haar cascade from {cascade_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces

def get_largest_face(faces):
    """
    Selects the largest face from a list of detected faces.

    Args:
        faces (list): A list of (x, y, w, h) tuples for detected faces.

    Returns:
        tuple: (x, y, w, h) for the largest face, or None if no faces are provided.
    """
    if not len(faces):
        return None
    
    largest_face = max(faces, key=lambda item: item[2] * item[3]) # item[2] is width, item[3] is height
    return largest_face

if __name__ == '__main__':
    # Example usage:
    # Create a dummy image for testing
    # In a real scenario, replace 'dummy_image.jpg' with an actual image path
    if not os.path.exists('dummy_image.jpg'):
        # Create a simple dummy image if it doesn't exist
        dummy_img_arr = cv2.UMat(np.zeros((200, 200, 3), dtype=np.uint8))
        # You might want to draw a face or use a known image for more robust testing
        cv2.imwrite('dummy_image.jpg', dummy_img_arr)

    # Attempt to download haarcascade if not present (requires internet)
    cascade_file = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_file):
        print(f"{cascade_file} not found. Attempting to download...")
        import urllib.request
        try:
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, cascade_file)
            print(f"Successfully downloaded {cascade_file}")
        except Exception as e:
            print(f"Error downloading {cascade_file}: {e}")
            print("Please ensure 'haarcascade_frontalface_default.xml' is available or OpenCV is correctly installed with Haar cascades.")
            exit()
            
    detected_faces = detect_faces('dummy_image.jpg', cascade_path=cascade_file)

    if detected_faces is None:
        print("Face detection failed critically (e.g. cascade not loaded).")
    elif not detected_faces:
        print("No faces detected in dummy_image.jpg")
    else:
        print(f"Detected {len(detected_faces)} faces in dummy_image.jpg.")
        largest = get_largest_face(detected_faces)
        if largest:
            x, y, w, h = largest
            print(f"Largest face at: x={x}, y={y}, w={w}, h={h}")

            # To visualize (optional, requires an image with a face)
            # img = cv2.imread('dummy_image.jpg')
            # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # cv2.imshow('Largest Face', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else: # Should not happen if detected_faces is not empty
             print("No faces found by get_largest_face, though detect_faces returned some.")

    # Clean up dummy image
    # if os.path.exists('dummy_image.jpg'):
    #     os.remove('dummy_image.jpg') 