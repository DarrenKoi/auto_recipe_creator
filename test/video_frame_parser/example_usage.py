#!/usr/bin/env python3
"""
Video Frame Parser 사용 예제

H200 클러스터를 활용한 동영상 프레임 추출 및 분석 예제 코드.
"""

from pathlib import Path

# 기본 사용법
def basic_usage():
    """기본 사용 예제"""
    from video_frame_parser import VideoFrameParser, VideoParserConfig

    # 설정 생성
    config = VideoParserConfig()
    config.extractor.frame_interval = 1.0  # 1초마다 프레임 추출

    # 파서 생성 및 초기화
    parser = VideoFrameParser(config)
    parser.initialize(use_gpu=False)

    # 단일 동영상 처리
    result = parser.process_video(
        "path/to/video.avi",
        save_to_db=True,
        save_frames_to_disk=False,
    )

    print(f"처리된 프레임: {result.processed_frames}/{result.total_frames}")
    print(f"처리 시간: {result.total_time_seconds:.2f}초")
    print(f"FPS: {result.frames_per_second:.1f}")

    # 리소스 정리
    parser.cleanup()


def h200_cluster_usage():
    """H200 클러스터 사용 예제"""
    from video_frame_parser import create_h200_optimized_parser

    # H200 최적화 파서 생성 (8 GPU)
    parser = create_h200_optimized_parser(
        num_gpus=8,
        mongo_uri="mongodb://localhost:27017"
    )

    # 여러 동영상 배치 처리
    video_paths = [
        "path/to/video1.avi",
        "path/to/video2.avi",
        "path/to/video3.avi",
    ]

    results = parser.process_videos_batch(
        video_paths,
        frame_interval=0.5,  # 0.5초마다 프레임 추출
        max_frames_per_video=100,
    )

    for result in results:
        print(f"Video {result.video_id}: {result.processed_frames} frames, "
              f"Success rate: {result.success_rate:.1%}")

    parser.cleanup()


def frame_extraction_only():
    """프레임 추출만 수행하는 예제"""
    from video_frame_parser import VideoFrameExtractor, ExtractorConfig

    config = ExtractorConfig(
        frame_interval=0.5,
        resize_width=640,
        resize_height=480,
        output_format="jpg",
        quality=85,
    )

    with VideoFrameExtractor(config) as extractor:
        # 동영상 열기
        metadata = extractor.open("path/to/video.avi")
        print(f"Video: {metadata.file_name}")
        print(f"Duration: {metadata.duration:.2f}s, FPS: {metadata.fps}")

        # 프레임 추출 및 저장
        output_dir = Path("./extracted_frames")
        for frame in extractor.extract_frames(max_frames=50):
            # 프레임 저장
            extractor.save_frame(frame, output_dir)
            print(f"Saved frame {frame.frame_number} at {frame.timestamp:.2f}s")


def similarity_search_example():
    """유사 프레임 검색 예제"""
    from video_frame_parser import VideoFrameParser, VideoParserConfig
    import numpy as np

    config = VideoParserConfig()
    parser = VideoFrameParser(config)
    parser.initialize()

    # 먼저 동영상 처리 (DB에 저장)
    parser.process_video("path/to/video.avi", save_to_db=True)

    # 쿼리 임베딩으로 유사 프레임 검색
    # (실제로는 분석된 프레임의 임베딩을 사용)
    query_embedding = np.random.randn(512).astype(np.float32)

    similar_frames = parser.search_similar_frames(
        query_embedding,
        top_k=10,
        threshold=0.7,
    )

    for item in similar_frames:
        print(f"Frame: {item['frame_id']}, Similarity: {item['similarity']:.3f}")

    parser.cleanup()


def custom_callback_example():
    """커스텀 콜백 사용 예제"""
    from video_frame_parser import VideoFrameParser, VideoParserConfig

    def on_frame_processed(result):
        """프레임 처리 완료 시 호출되는 콜백"""
        if result.status.value == "completed":
            print(f"✓ Frame {result.frame_id} processed in {result.processing_time_ms:.1f}ms")
        else:
            print(f"✗ Frame {result.frame_id} failed: {result.error_message}")

    config = VideoParserConfig()
    parser = VideoFrameParser(config)
    parser.initialize()

    # 콜백과 함께 처리
    result = parser.process_video(
        "path/to/video.avi",
        callback=on_frame_processed,
    )

    parser.cleanup()


def keyframe_extraction_example():
    """키프레임만 추출하는 예제"""
    from video_frame_parser import VideoFrameExtractor, ExtractorConfig

    config = ExtractorConfig()

    with VideoFrameExtractor(config) as extractor:
        extractor.open("path/to/video.avi")

        # 키프레임만 추출 (씬 변화 감지)
        keyframes = list(extractor.extract_keyframes())

        print(f"Detected {len(keyframes)} keyframes (scene changes)")
        for kf in keyframes:
            print(f"  - Frame {kf.frame_number} at {kf.timestamp:.2f}s")


if __name__ == "__main__":
    print("Video Frame Parser 사용 예제")
    print("=" * 50)
    print("\n이 파일은 예제 코드입니다.")
    print("실제 사용 시 video_path를 실제 동영상 경로로 변경하세요.\n")

    # 통계 출력 예제
    from video_frame_parser import VideoFrameParser

    parser = VideoFrameParser()
    print("Parser 초기화 전 상태:")
    print(parser.get_stats())
