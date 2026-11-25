# LoveDA notes

## Data
- Source: LoveDA dataset (Urban and Rural scenes), aerial imagery
- Local copy: data/Val/{Urban,Rural}/{images_png,masks_png} (not committed, in .gitignore)
- Colab copy: /content/data/Train (re-download each session)
- Images: 1024x1024 RGB. Masks: 1024x1024, single channel, uint8
- Val Urban counts: 677 images, 677 masks

## Class numbers in the masks
| Number | Meaning | How I know |
|---|---|---|
| 0 | no-data (unlabelled pixels) | appears in a few masks; official LoveDA numbering |
| 1 | background | orange strips beside the road in the plot |
| 2 | building | appears in most masks except farmland-only ones |
| 3 | road | red band exactly on the road in the plot |
| 4 | water | purple on river and canal shapes |
| 5 | barren | not checked yet |
| 6 | forest | not checked yet |
| 7 | agriculture | gray, 83% of the first mask |

## Observations
- First Val Urban mask: values [1 3 4 7], counts [19074 49500 109563 870439]. Mostly farmland, road about 4.7%, no buildings.
- Masks are coarse human drawings, so edges will be rough.
- Some crops have no buildings or roads at all (e.g. 3519.png is all water).
- Roads and buildings are small shares of the pixels, so training needs class weights.

## Plan for training
- Merge 8 classes into 3: 0 = everything else, 1 = building (2), 2 = road (3)
- Start with Urban scenes only
- Random 256x256 crops with flips
- Measure IoU per class on Val