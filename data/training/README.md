# Training Data

Place your training images in the `images/` folder and update `annotations.json`.

## Annotation Format

```json
[
  {
    "image": "images/tractor_field_001.jpg",
    "boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2]]
  }
]
```

- **image**: Relative path to the image file
- **boxes**: Array of bounding boxes in format [x1, y1, x2, y2]
  - (x1, y1): Top-left corner
  - (x2, y2): Bottom-right corner

## Example

If you have a person at coordinates:
- Top-left: (100, 150)
- Bottom-right: (200, 400)

Your annotation would be: `[100, 150, 200, 400]`

## Tools for Annotation

You can use these tools to create annotations:
- [LabelImg](https://github.com/heartexlabs/labelImg) - Desktop tool
- [CVAT](https://www.cvat.ai/) - Web-based annotation
- [Roboflow](https://roboflow.com/) - Cloud-based with export options

After annotating, convert to the JSON format above.
