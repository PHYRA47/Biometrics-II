function processedImage = preProcess(imageFilename)

    % Step 1: Read the input image
    img = imread(imageFilename);
    figure; imshow(img); title('Original Image');

    % Step 2: Apply median filtering
    filterSize = 4;
    filtImg = medfilt2(img, [filterSize filterSize]);
    figure; imshow(filtImg); title('Filtered Image');

    % Step 3: Convert to binary image using Otsu's thresholding
    BW = imbinarize(filtImg, graythresh(img));
    
    % Step 4: Invert the binary image
    BW_inv = ~BW;
    figure; imshow(BW_inv); title('Inverted Binary Image');

    % Step 5: Fill holes in the binary image
    BW_filled = imfill(BW_inv, "holes");
    figure; imshow(BW_filled); title('Filled Binary Image');

    % Step 6: Apply morphological opening (remove small objects)
    BW_open = imopen(BW_filled, strel('disk', 2));
    figure; imshow(BW_open); title('After Opening Operation');

    % Step 7: Apply morphological closing (close small gaps)
    BW_close = imclose(BW_open, strel('disk', 2));
    figure; imshow(BW_close); title('After Closing Operation');

    % Step 8: Remove small connected components
    BW_area_open = bwareaopen(BW_close, 80, 8);
    figure; imshow(BW_area_open); title('Final Processed Image');

    % Return the final processed binary image
    processedImage = BW_area_open;
end