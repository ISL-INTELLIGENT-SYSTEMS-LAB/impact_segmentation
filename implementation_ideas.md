## This is brainstorming on how to update the current pipeline

Currently we pass direct cordinates of objects with bounding boxes to dinov2. This is our reference image, but it isn't practical, because it can only determine that objects are in the same class, not necessarily that this is the same object as the first picture

We need to fix this

## IDEAs

- passing to dinov2 all the objects in the pictures bounding boxes and then seeing if dinov2 can detect the same objects in another picture based of cosine similarity scores. This kinda might deliver the same problems. I am not sure I will research this. 

- Going back to ASTR, ASTR has spot guided identification, even though bounding boxes aren't the spots this could be a valid way of actually finding the same objects through pictures, we would just have to get rid of the dinov2 pipeline. I believe this could work well because it isn't just relying on sam3 with this work.

- add in the ASTR, we need to implement it
