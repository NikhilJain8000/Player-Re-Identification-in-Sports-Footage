
# Technical Report: Player Re-Identification in Sports Footage

**Author:** Nikhil Jain  
**Date:** June 27, 2025  
**Assignment:** Liat.ai AI Internship - Player Re-Identification

## 1. Executive Summary

When I first approached this challenge, I knew that tracking players in sports footage would require more than just a simple detection algorithm. The real challenge lies in maintaining consistent player identities when they disappear behind other players or run out of frame - something that happens constantly in dynamic sports scenarios.

My solution implements a robust player re-identification system that successfully tracks multiple players throughout the video, maintaining their unique IDs even after temporary occlusions. I achieved this by developing a two-stage matching approach that cleverly combines motion-based tracking with appearance-based re-identification.

## 2. Approach and Methodology

### 2.1 Overall Architecture

I structured my solution around three core components that work together seamlessly:

1. **Detection Module** (YOLOv11): This handles the initial spotting of players, goalkeepers, and referees in each frame
2. **Feature Extraction Module** (ResNet50): My choice for generating unique visual "fingerprints" for each player
3. **Tracking Module** (Hungarian Algorithm): The brain that maintains consistent player identities across frames

### 2.2 Two-Stage Matching Strategy

The breakthrough in my implementation came when I realized that relying on just one matching method wasn't enough. Here's how I solved it:

#### Stage 1: Motion-Based Tracking (IOU Matching)
- **Purpose**: Keep players tracked between consecutive frames
- **Method**: I calculate how much bounding boxes overlap between frames
- **Threshold**: IOU > 0.3 means it's likely the same player
- **Why this works**: Players don't teleport - they move predictably frame-to-frame

#### Stage 2: Appearance-Based Re-Identification
- **Purpose**: Find players again after they've been hidden or left the frame
- **Method**: Compare visual similarity using deep learning features
- **Model**: ResNet50 pre-trained on ImageNet (removed the classification layer)
- **Output**: Each player gets a unique 2048-dimensional "signature"
- **Threshold**: Similarity > 0.4 triggers a match

### 2.3 Hungarian Algorithm for Optimal Assignment

I chose the Hungarian algorithm because greedy matching kept failing me. This algorithm ensures:
- No two detections fight over the same ID
- We get the globally best assignment, not just locally good ones
- Players with similar jerseys don't swap IDs

### 2.4 Track Management

My tracking system keeps tabs on players intelligently:
- **Smart Updates**: I update embeddings slowly (α=0.9) to avoid sudden changes
- **Memory**: Tracks remember players for 60 frames (~2 seconds) after they disappear
- **Stability**: New IDs only appear after 1-2 consistent detections

## 3. Technical Implementation Details

### 3.1 YOLO Integration

The first "aha" moment came when I realized the model was detecting the ball as class 0:

```python
# My fix: Only detect humans (classes 1, 2, 3), ignore the ball (class 0)
results = detector(frame, classes=[1, 2, 3], verbose=False)
```

### 3.2 Feature Extraction Pipeline

Here's my step-by-step process for creating player signatures:
1. Extract the player using their bounding box
2. Convert from OpenCV's BGR to RGB (this tripped me up initially!)
3. Resize to 224×224 (ResNet's expected input)
4. Normalize using ImageNet's mean and std
5. Get the 2048-dimensional embedding

### 3.3 Optimization Techniques

I implemented several optimizations to make the system practical:
- Automatic GPU detection and usage when available
- Vectorized operations wherever possible
- Skip processing for invalid or tiny crops
- Batch processing ready (for future improvements)

## 4. Challenges Encountered and Solutions

### 4.1 Challenge: The Frustrating Frame-to-Frame ID Switching

**What happened**: My first version gave everyone new IDs every single frame!  
**My solution**: I added IOU matching as the first stage. If a player is roughly in the same spot, it's probably the same person.

### 4.2 Challenge: The Jersey Problem

**What happened**: Players on the same team kept swapping IDs because they look identical  
**My solution**: 
- Lowered the similarity threshold after extensive testing
- Added position information to break ties
- Implemented smooth embedding updates to maintain consistency

### 4.3 Challenge: The Ball Was Being Tracked as a Player

**What happened**: For the longest time, "Player 0" was just the ball bouncing around!  
**My solution**: Properly filtered the YOLO classes to exclude class 0 (ball)

### 4.4 Challenge: Players Disappearing Behind Each Other

**What happened**: Occlusions caused ID loss and reassignment  
**My solution**: 
- Keep "ghost tracks" for 60 frames
- Trust the Hungarian algorithm to sort out the mess when they reappear

## 5. Results and Performance

### 5.1 What My System Can Do

After all the iterations and debugging, I'm proud to say it:
- ✓ Maintains consistent IDs across the entire video
- ✓ Successfully re-identifies players after they return to view
- ✓ Handles 15+ players simultaneously without breaking a sweat
- ✓ Robust to camera movement and zoom changes
- ✓ Runs at near real-time speeds on a decent GPU

### 5.2 Visual Feedback

I added several visualization features that really help understand what's happening:
- Each player gets a unique, consistent color
- Confidence scores show when someone is re-identified vs newly detected
- Frame counter and statistics overlay
- The ball is highlighted separately (but not tracked)

## 6. Future Improvements

If I had more time (and maybe during the internship!), here's what I'd love to implement:

### 6.1 Smarter Feature Extraction
- Fine-tune the ResNet on actual soccer/sports data
- Try to read jersey numbers as an additional feature
- Experiment with newer architectures like Vision Transformers

### 6.2 Motion Prediction
- Add Kalman filters to predict where players will be
- Use optical flow for smoother tracking
- Implement physics-based constraints

### 6.3 Game Understanding
- Automatically identify teams by jersey colors
- Classify player positions (goalkeeper, defender, etc.)
- Detect game events (goals, fouls, etc.)

### 6.4 Speed Optimizations
- Convert to TensorRT for production deployment
- Implement proper multi-threading
- Add dynamic quality adjustment based on GPU load

### 6.5 Analytics Dashboard
- Real-time player statistics
- Heat maps showing player movement
- Ball possession percentages
- Formation analysis

## 7. Conclusion

Building this player re-identification system was both challenging and incredibly rewarding. The journey from tracking just the ball as "Player 0" to successfully maintaining 15+ player identities taught me valuable lessons about real-world computer vision challenges.

The two-stage matching approach I developed provides an elegant balance between computational efficiency and tracking accuracy. By combining the immediate reliability of position-based tracking with the long-term robustness of appearance-based re-identification, the system handles the complexities of real sports footage.

I'm particularly proud of how the modular architecture turned out - each component has a clear responsibility and can be improved independently. This design philosophy, combined with comprehensive documentation, ensures that the system is not just a proof-of-concept but a foundation for production-ready sports analytics.

## 8. Technical Skills Demonstrated

Through this project, I've had the opportunity to showcase:

- **Computer Vision**: From basic detection to complex re-identification strategies
- **Deep Learning**: Practical application of pre-trained models and transfer learning
- **Algorithm Design**: Implementing and adapting the Hungarian algorithm for optimal matching
- **Software Engineering**: Writing clean, modular, well-documented code
- **Problem Solving**: Debugging tricky issues like negative strides and class confusion
- **Performance Optimization**: Balancing accuracy with speed using GPU acceleration

Working on this assignment reinforced my passion for applying AI to sports analytics. The combination of technical challenges and real-world applications is exactly what draws me to this field. I'm excited about the possibility of joining Liat.ai and contributing to cutting-edge sports technology solutions.

Thank you for this opportunity to demonstrate my skills. I look forward to discussing how I can contribute to your team!
