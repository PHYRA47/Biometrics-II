function [imFinalResult] = watershedSegmentation(imPreProcessed)
    % Step 1: Compute the distance transform and identify extended minima
    imDistTransform = -bwdist(~imPreProcessed);
    % figure; imshow(imDistTransform, []); title('Distance Transform');

    minimaMask = imextendedmin(imDistTransform, 2);
    % figure; imshow(minimaMask); title('Extended Minima Mask');

    imImposeMin = imimposemin(imDistTransform, minimaMask);
    % figure; imshow(imImposeMin, []); title('Imposed Minima');

    % Step 2: Perform watershed segmentation
    imWater = watershed(imImposeMin);
    % figure; imshow(label2rgb(imWater)); title('Watershed Segmentation');

    % Step 3: Final Result
    imFinalResult = imPreProcessed;
    imFinalResult(imWater == 0) = 0;
    % figure; imshow(imFinalResult); title('Final Result');
end