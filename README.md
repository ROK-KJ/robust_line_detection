# robust_line_detection
CNRLab Robust Line Detection by yeseul Jeong


이 레파지토리는 아직 개발중입니다. 


## 목차

- [프로젝트 설명 및 개요](#프로젝트-설명-및-개요)
- [사용 방법](#사용-방법)
- [prerequisites](#prerequisites)
  

## 프로젝트 설명 및 개요

SLAM에 사용하기 위한 강인한 Line 추출 및 추적 

>
> 1. edge detection 방식을 결정하기 위한 cost function 정의
> 2. Grid search 방식의 best edge detection parameter 추출 및 edge detection 수행
> 3. contour 길이 및 개수를 이용한 edge 단순화
> 4. 확률적 HoughLine을 사용한 vertical line 추출 
> 5. line의 endpoint를 이용하여 추적 포인트 생성 및 추적 반복
>

### 결과 사진 예시 

![프로젝트 이미지](https://file.notion.so/f/f/33ab5710-0e62-4945-b834-ff76ea81a48f/d75e63d0-1264-4e59-a9e6-d1cdb2e628ef/Untitled.png?id=3e50fa97-c4d1-4f11-8c26-42d295caf704&table=block&spaceId=33ab5710-0e62-4945-b834-ff76ea81a48f&expirationTimestamp=1708768800000&signature=Ae16QAanzErpU0RVeS2Fozri-eNvu9KY6hUx3kq_vpI&downloadName=Untitled.png)


## 사용 방법

git clone 후, `run.py` 실행 

실행 시 입력 파라미터로 실행할 video path 입력

### Example
```
python3 run.py --input test.mp4
```


## prerequisites

```
opencv-python
scikit-learn
```
