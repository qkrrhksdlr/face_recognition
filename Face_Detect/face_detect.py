import cv2
import os
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw

# 입력, 출력 이미지
input_file = 'image/input.jpg'
output_file = 'image/output.jpg'

# 실시간 사진 찍기
def capture():
    # 카메라 켜기
    camera = cv2.VideoCapture(0)
    # 화면 크기 설정
    camera.set(3, 1000)
    camera.set(4, 1000)

    while True:
        # fame 별로 capture
        ret, frame = camera.read()
        # 화면에 출력
        cv2.imshow('frame', frame)
        # image 폴더에 jpg 파일로 저장
        cv2.imwrite(input_file, frame)

        # 'q' 누르면 화면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 프로그램 clear 하고 windows 닫기
    camera.release()
    cv2.destroyAllWindows()

# 얼굴 감지하기
def detect_face(face_file):
    # 서비스 객체 만들기
    client = vision.ImageAnnotatorClient()
    # 이미지 읽기
    content = face_file.read()
    image = types.Image(content=content)
    # 이미지에 대한 얼굴을 리스트로 반환
    return client.face_detection(image=image).face_annotations

# 하이라이트 처리
def highlight_faces(image, faces, output_filename):
    im = Image.open(image)
    draw = ImageDraw.Draw(im)
    # 폰트와 사이즈를 구체화
    for face in faces:
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        # 인식된 얼굴의 정확도 나타내기
        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y - 30),
                  str(format(face.detection_confidence, '.3f')) + '%',
                  fill='#FF0000')
    # 새로운 파일을 저장
    im.save(output_filename)

# 얼굴인식 실행
def main(input_filename, output_filename):
    with open(input_filename, 'rb') as image:
        # 얼굴 감지하기
        faces = detect_face(image)
        # 감지된 얼굴수 출력
        print('Found {} face{}'.format(len(faces), '' if len(faces) == 1 else 's'))
        print(faces)
        print('Writing to file {}'.format(output_filename))
        # 파일포인터 리셋
        image.seek(0)
        # 하이라이트 처리
        highlight_faces(image, faces, output_filename)

# 함수 실행
capture()
main(input_file, output_file)

