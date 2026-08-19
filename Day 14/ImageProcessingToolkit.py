import cv2
import numpy as np
import os

image = None
original_image = None

while True:
    print("\n========== Image Processing Toolkit ==========")
    print("1. Load Image")
    print("2. Convert to Grayscale")
    print("3. Resize Image")
    print("4. Rotate Image")
    print("5. Flip Image")
    print("6. Crop Image")
    print("7. Draw Shapes")
    print("8. Add Custom Text")
    print("9. Save Image")
    print("10. Adjust Brightness")
    print("11. Adjust Contrast")
    print("12. Convert BGR to RGB")
    print("13. Compare Original and Processed")
    print("0. Exit")

    choice = input("\nEnter your choice: ")


    # Load Image
    if choice == "1":

        images = [
            r"D:\Internship\Day 14\Sample Input Images\landscape.jpg",
            r"D:\Internship\Day 14\Sample Input Images\person.jpg",
            r"D:\Internship\Day 14\Sample Input Images\vehicle.jpg",
            r"D:\Internship\Day 14\Sample Input Images\document.jpg",
            r"D:\Internship\Day 14\Sample Input Images\object.jpg"
        ]

        print("\nSelect Image:")
        for i, img_path in enumerate(images):
            print(f"{i+1}. {os.path.basename(img_path)}")

        img_choice = int(input("Choose image: "))

        image = cv2.imread(images[img_choice-1])

        if image is None:
            print("Image not found.")
        else:
            original_image = image.copy()
            print("Image loaded successfully.")

            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    # Grayscale
    elif choice == "2":

        if image is None:
            print("Load an image first.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Grayscale", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Resize
    elif choice == "3":

        if image is None:
            print("Load an image first.")
            continue

        width = int(input("Enter new width: "))
        height = int(input("Enter new height: "))

        image = cv2.resize(image, (width, height))

        cv2.imshow("Resized", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Rotate
    elif choice == "4":

        if image is None:
            print("Load an image first.")
            continue

        print("1. 90 Clockwise")
        print("2. 180")
        print("3. 270")

        r = input("Choose rotation: ")

        if r == "1":
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        elif r == "2":
            image = cv2.rotate(image, cv2.ROTATE_180)

        elif r == "3":
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imshow("Rotated", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Flip
    elif choice == "5":

        if image is None:
            print("Load an image first.")
            continue

        print("1. Horizontal")
        print("2. Vertical")

        f = input("Choose flip: ")

        if f == "1":
            image = cv2.flip(image, 1)

        elif f == "2":
            image = cv2.flip(image, 0)

        cv2.imshow("Flipped", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Crop
    elif choice == "6":

        if image is None:
            print("Load an image first.")
            continue

        y1 = int(input("Start Row: "))
        y2 = int(input("End Row: "))
        x1 = int(input("Start Column: "))
        x2 = int(input("End Column: "))

        image = image[y1:y2, x1:x2]

        cv2.imshow("Cropped", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.rectangle(image, (top left corner), (bottom right corner), (0,255,0), 3)
        # cv2.circle(image, (center), radius, (255,0,0), 3)
        # cv2.line(image, (starting point), (ending point), (0,0,255), 3)

    # Draw Shapes
    elif choice == "7":

        if image is None:
            print("Load an image first.")
            continue

        draw = image.copy()

        cv2.rectangle(draw, (50,50), (250,250), (0,255,0), 3)
        cv2.circle(draw, (400,150), 60, (255,0,0), 3)
        cv2.line(draw, (50,350), (500,350), (0,0,255), 3)

        image = draw

        cv2.imshow("Shapes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# cv2.putText(
#     image,
#     text,
#     position,
#     font,
#     fontScale,
#     color,
#     thickness
# )
    # Add Text
    elif choice == "8":

        if image is None:
            print("Load an image first.")
            continue

        text = input("Enter text: ")

        cv2.putText(
            image,
            text,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Text", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Save Image
    elif choice == "9":

        if image is None:
            print("Load an image first.")
            continue

        os.makedirs("output", exist_ok=True)

        filename = input("Enter filename: ")

        if not filename.endswith((".jpg",".png",".jpeg")):
            filename += ".jpg"

        path = os.path.join("output", filename)

        cv2.imwrite(path, image)

        print("Image saved successfully.")


    # Brightness
    elif choice == "10":

        if image is None:
            print("Load an image first.")
            continue

        value = int(input("Brightness value: "))

        bright = cv2.convertScaleAbs(image, alpha=1, beta=value)

        cv2.imshow("Brightness", bright)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Contrast
    elif choice == "11":

        if image is None:
            print("Load an image first.")
            continue

        value = float(input("Contrast value (1-3): "))

        contrast = cv2.convertScaleAbs(image, alpha=value, beta=0)

        cv2.imshow("Contrast", contrast)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # BGR to RGB
    elif choice == "12":

        if image is None:
            print("Load an image first.")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imshow("BGR Original", image)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow("RGB Converted", rgb)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Compare
    elif choice == "13":

        if original_image is None or image is None:
            print("Load an image first.")
            continue

        img1 = cv2.resize(original_image, (400,400))
        img2 = cv2.resize(image, (400,400))

        comparison = np.hstack((img1,img2))

        cv2.imshow("Original | Processed", comparison)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Exit
    elif choice == "0":

        print("Thank you!")
        break


    else:
        print("Invalid Choice.")