//go:build integration

package integration

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/lwlee2608/aiwire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const imageGenModel = "google/gemini-2.5-flash-image"

// saveImage writes data to /tmp/<name>.<ext> and logs the path. ext falls back
// to "bin" when the container is unrecognized.
func saveImage(t *testing.T, name, ext string, data []byte) {
	t.Helper()
	if ext == "" {
		ext = "bin"
	}
	path := filepath.Join("/tmp", name+"."+ext)
	require.NoError(t, os.WriteFile(path, data, 0o644))
	t.Logf("Saved image to %s", path)
}

// imageMagic returns a short label for a decoded image's container, or "" if unrecognized.
func imageMagic(data []byte) string {
	switch {
	case bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n")):
		return "png"
	case bytes.HasPrefix(data, []byte("\xff\xd8\xff")):
		return "jpeg"
	case bytes.HasPrefix(data, []byte("GIF8")):
		return "gif"
	case bytes.HasPrefix(data, []byte("RIFF")) && len(data) > 11 && bytes.Equal(data[8:12], []byte("WEBP")):
		return "webp"
	default:
		return ""
	}
}

func TestOpenRouter_ImageGeneration(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")

	resp, err := service.GenerateImage(context.Background(), aiwire.ImageOption{
		Model:       imageGenModel,
		Prompt:      "A simple solid red square on a white background.",
		AspectRatio: "1:1",
	})
	require.NoError(t, err)
	require.NotEmpty(t, resp.Images, "expected at least one generated image")

	t.Logf("Text: %s", resp.Text)
	t.Logf("Provider: %s", resp.Provider)
	t.Logf("Images: %d", len(resp.Images))
	logUsage(t, resp.Usage)

	mime, data, err := resp.Images[0].Decode()
	require.NoError(t, err)
	assert.NotEmpty(t, data)
	kind := imageMagic(data)
	t.Logf("First image: mime=%s bytes=%d kind=%s", mime, len(data), kind)
	assert.NotEmpty(t, kind, "decoded data should be a recognizable image")
	saveImage(t, "aiwire_image_generation", kind, data)
}

func TestOpenRouter_ImageEditing(t *testing.T) {
	service := aiwire.NewOpenAIService(keyOrSkip(t, "OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1")

	// ocrBase64PNG is embedded in ocr_test.go (same package): a 300x80 PNG.
	resp, err := service.GenerateImage(context.Background(), aiwire.ImageOption{
		Model:  imageGenModel,
		Prompt: "Add a bright yellow border around this image.",
		Images: []aiwire.ImageInput{
			aiwire.ImageInputFromBytes("image/png", ocrBase64PNG),
		},
	})
	require.NoError(t, err)
	require.NotEmpty(t, resp.Images, "expected at least one edited image")

	t.Logf("Provider: %s", resp.Provider)
	logUsage(t, resp.Usage)

	mime, data, err := resp.Images[0].Decode()
	require.NoError(t, err)
	assert.NotEmpty(t, data)
	kind := imageMagic(data)
	t.Logf("Edited image: mime=%s bytes=%d kind=%s", mime, len(data), kind)
	assert.NotEmpty(t, kind, "decoded data should be a recognizable image")
	saveImage(t, "aiwire_image_editing", kind, data)
}
