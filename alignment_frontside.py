import numpy as np
import cv2

def segment_card(img):
    """
    Segment the area containing the card in the image using the color.
    The Vietnamese ID card has color green, so we use thresholding with the hue in each pixel.
    
    Parameters:
    img: The input colored image containing the ID card in RGB channel last format.

    Return:
    mask: The binary image with the same size as img indicating the card pixels
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask_noisy = np.logical_and(hsv[:,:,0] >= 40, hsv[:,:,0] < 100)
    # Remove the black and white area where hue might accidentally fall into the range
    mask_noisy = np.logical_and(mask_noisy, hsv[:,:,1] >= 20)
    mask_noisy = np.logical_and(mask_noisy, hsv[:,:,2] >= 20)
    mask_noisy = (mask_noisy * 255).astype('uint8')
    mask = cv2.morphologyEx(mask_noisy, cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                        iterations=30)      
    return mask


def get_boundary(mask):
    """
    Get 4 sides of the card from the mask.

    Parameters:
    mask: The binary image indicating the area containing the card in the image

    Return: htop, hbottom, vleft, vright - 4 lines corresponding to 4 sides
    Each line is represented as a tuple (rho, theta) for the line x*cos(theta) + y*sin(theta) = rho
    """
    # Get the points on the boundary by finding contour
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    boundaries = np.zeros_like(mask, dtype='uint8')
    cv2.drawContours(boundaries, contours, -1, 255)

    # Use Hough transform to find the lines fitted to the contour points
    lines = cv2.HoughLines(boundaries, 10, np.pi/60, 100)
    lines = lines.reshape(-1, 2)

    # Categorize horizontal vs vertical line based on theta
    hlines = []
    vlines = []
    for line in lines:
        if np.pi/4 <= line[1] <= 3*np.pi/4:
            hlines.append(line)
        else:
            vlines.append(line)
    hlines = np.array(hlines)
    vlines = np.array(vlines)

    # Categorize top-bottom and left-right based on rho
    htop = None
    hbottom = None
    for hline in hlines:
        if hline[0] < np.mean(hlines, axis=0)[0]:
            if htop is None:
                htop = hline
        else:
            if hbottom is None:
                hbottom = hline

    vleft = None
    vright = None
    for vline in vlines:
        if abs(vline[0]) < np.mean(np.abs(vlines), axis=0)[0]:
            if vleft is None:
                vleft = vline
        else:
            if vright is None:
                vright = vline
    return htop, hbottom, vleft, vright


def get_corner(htop, hbottom, vleft, vright):
    """
    Compute the 4 corners given the 4 sides

    Parameters:
    htop, hbottom, vleft, vright: 4 sides, each formatted as (rho, theta)

    Returns:
    topleft, topright, bottomleft, bottomright: 4 corners
    """
    def find_intersect(hline, vline):
        A = np.array([[np.cos(hline[1]), np.sin(hline[1])], [np.cos(vline[1]), np.sin(vline[1])]])
        b = np.array([hline[0], vline[0]])
        return np.linalg.solve(A, b)

    topleft = find_intersect(htop, vleft)
    topright = find_intersect(htop, vright)
    bottomleft = find_intersect(hbottom, vleft)
    bottomright = find_intersect(hbottom, vright)

    return topleft, topright, bottomleft, bottomright


def warp_img(img, region, target_dsize):
    """
    Cut a quadrilateral part of the image and warp to a new image

    Parameters:
    img: The original image
    region: Tuple of 4 points indicating 4 vertices topleft, topright, bottomleft, bottomright. Each point is a pair (x, y)
    target_dsize: (width, height) tuple of the targeted image size

    Returns:
    img_warped: The warped image
    """
    keypoints = np.array(region)
    w, h = target_dsize
    targets = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(keypoints, targets)
    img_warped = cv2.warpPerspective(img, M, target_dsize)
    return img_warped


def align_card(img, dsize=(800, 500)):
    """
    Extract the region containing the ID card in the image and warp it to a new image

    Parameters:
    img: The image containing the ID card with format RGB channel last
    dsize: The size of the resulting image formatted as (width, height)

    Returns:
    img_aligned: The image containing only the card area
    """
    # Rescale to standardize the image size
    scale_factor = 1000 / img.shape[1]
    img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    # Add border to deal with the cases where the sides touch the image boundary
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, 0)
    mask = segment_card(img)
    boundary_lines = get_boundary(mask)
    corners = get_corner(*boundary_lines)
    img_aligned = warp_img(img, corners, dsize)
    return img_aligned