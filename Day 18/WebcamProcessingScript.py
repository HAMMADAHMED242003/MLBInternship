import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(
    'webcam_output.avi',
    fourcc,
    20.0,
    (640,480)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,100,200)

    cv2.imshow("Original",frame)

    cv2.imshow("Processed",edges)

    out.write(cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR))

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()

out.release()

cv2.destroyAllWindows()