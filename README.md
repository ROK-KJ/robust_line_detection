# robust_line_detection
CNRLab Robust Line Detection by yeseul Jeong


이 레파지토리는 아직 개발중입니다. 

> <b>(24.03.05) [update ROS-version]</b> image topic subscribe 하여 결과 출력 형식 구현 ▶▶ Line_Detection_ROS 파일에 따로 업로드 하였음. 
> <b>(24.03.12) [update cost function, line algorithm]</b> cost function update (figure of merit ▶ displacement of edge images, line merge function added)


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

[추후 업로드] 

## 사용 방법

### [video version]

git clone 후, Line_Detection directory의 `run.py` 실행 

실행 시 입력 파라미터로 실행할 video path 혹은 camera number (0, 1, ...) 입력

#### Video Input Example
```
python3 run.py --input test.mp4
```

#### Camera Input Example
```
python3 run.py --input 0
```


### [ROS application version]

git clone 후, Line_Detection directory의 `run.py` 실행 

`"/camera/rgb/image_raw"` topic을 subscribe하여 로직이 실행됨 

토픽명 변경하고자 할 시 LinePointTracker의 `self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.tracking_callback)` 부분 수정 

cost function 실행시간 이슈로 최초 실행 시 parameter를 저장할 `.yaml` 파일 생성 후 로드하여 사용하도록 수정 (.yaml 파일이 존재하는 경우에만 자동 로드 됨)


## prerequisites

```
opencv-python
scikit-learn
```
