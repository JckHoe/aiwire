package aiwire

import (
	"context"
	"encoding/base64"
)

// VideoGeneration generates videos from a text prompt and optional frame
// images (image-to-video). It is separate from [ImageGeneration] and
// [Completion] because video generation is asynchronous: OpenRouter's /videos
// API submits a job and the result is polled until completion.
type VideoGeneration interface {
	GenerateVideo(ctx context.Context, opt VideoOption) (VideoResponse, error)
}

// VideoFrameType selects which end of the clip a frame image anchors.
type VideoFrameType string

const (
	VideoFrameFirst VideoFrameType = "first_frame"
	VideoFrameLast  VideoFrameType = "last_frame"
)

// VideoOption configures a video-generation request.
type VideoOption struct {
	Model       string
	Prompt      string
	FrameImages []VideoFrameImage // optional first/last frames for image-to-video
	Duration    int               // clip length in seconds
	AspectRatio string            // e.g. "16:9", "9:16", "1:1"
	Resolution  string            // e.g. "1080p"
	ConfigExtra map[string]any    // extra top-level knobs (e.g. cfg_scale)
}

// VideoFrameImage is a source frame supplied for image-to-video generation.
type VideoFrameImage struct {
	URL       string         // data URL ("data:image/png;base64,...") or a remote URL
	FrameType VideoFrameType // defaults to first_frame
}

// VideoFrameFromBytes builds a VideoFrameImage as a base64 data URL.
func VideoFrameFromBytes(mimeType string, data []byte, frameType VideoFrameType) VideoFrameImage {
	return VideoFrameImage{
		URL:       "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data),
		FrameType: frameType,
	}
}

// VideoResponse is the result of a completed video-generation request.
type VideoResponse struct {
	Videos   []GeneratedVideo
	Provider string
	Usage    Usage
}

// GeneratedVideo is one video emitted by a video-generation model.
type GeneratedVideo struct {
	URL string // remote URL to the rendered clip
}
