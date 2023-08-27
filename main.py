import pygame
import pygame.camera
import numpy as np
import os
import random
import cv2
from tryout import MODEL


def Image2Binary(path):
    path = os.path.join("images", path)
    # Load the image
    image = cv2.imread(path)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (using a global threshold value of 230)
    _, image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)

    # Define the structuring element for morphological operations
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Erosion
    image = cv2.erode(image, structuring_element, iterations=1)

    # Dilation
    image = cv2.dilate(image, structuring_element, iterations=1)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image


def Numpy2Pygame(img, w, h):
    img = pygame.surfarray.make_surface(img)
    w = w / 100
    h = h / 100
    img = pygame.transform.scale(img, (int(w * 20), int(w * 20)))
    img = pygame.transform.rotate(img, 90 * 3)
    img = pygame.transform.flip(img, True, False)
    return img


def loadImage(path, w, h):
    img = Image2Binary(path)
    img = Numpy2Pygame(img, w, h)
    return img

model = MODEL()
images = random.sample(os.listdir('images'), 5)
index = 0

# Initialize Pygame
pygame.init()
pygame.camera.init()

# Get the list of available cameras
cam_list = pygame.camera.list_cameras()

if not cam_list:
    print("No cameras found.")
    exit()

# Connect to the first available camera and set resolution
width, height = 640, 480
camera = pygame.camera.Camera(cam_list[0], (width, height))

font = pygame.font.Font('RussoOne-Regular.ttf', 140)
WinFont = pygame.font.Font('RussoOne-Regular.ttf', 190)
Group1Score, Group2Score = 0, 0

startFont1 = pygame.font.Font('RussoOne-Regular.ttf', 100)
startFont2 = pygame.font.Font('RussoOne-Regular.ttf', 40)

# Create a Pygame display window
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Webcam Viewer")

width, height = screen.get_width(), screen.get_height()

target_image = loadImage(images[index], width, height)

pygame.display.update()

timePerStep = 10
flash = 0
stop = False
GroupTurn = 0
G1CORR, G2CORR = 0, 0

gameTimer = pygame.time.get_ticks()

Mode = "open"  # open / play / win

try:
    camera.start()

    while True:

        if Mode == "open":
            gameTimer = pygame.time.get_ticks()
            elapsed_seconds = 0
            # Capture a frame from the camera
            image = camera.get_image()

            # Scale the image to fit the screen resolution
            image = pygame.transform.scale(image, (width, height))

            # Flip the Image
            image = pygame.transform.flip(image, True, False)

            # Draw the Cam image on the screen
            screen.blit(image, (0, 0))

            line1 = "Welcome to FUCKakte!"
            line2 = "Press P to start"

            line1_text = startFont1.render(line1, True, (89, 215, 99))
            line2_text = startFont2.render(line2, True, (31, 171, 95))

            line1_rect = line1_text.get_rect(center=(width / 2, height / 2 - line1_text.get_height() / 2))
            line2_rect = line2_text.get_rect(center=(width / 2, height / 2 + line2_text.get_height() / 2))

            screen.blit(line1_text, line1_rect)
            screen.blit(line2_text, line2_rect)

            pygame.display.flip()

            # Check for a quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    camera.stop()
                    pygame.quit()
                    exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        camera.stop()
                        pygame.quit()
                        exit()

                    if event.key == pygame.K_p:
                        Mode = "play"
                        continue

            continue
        elif Mode == "win":
            # Capture a frame from the camera
            image = camera.get_image()

            # Scale the image to fit the screen resolution
            image = pygame.transform.scale(image, (width, height))

            # Flip the Image
            image = pygame.transform.flip(image, True, False)

            # Draw the Cam image on the screen
            screen.blit(image, (0, 0))

            # Show the score
            G1 = font.render(f"{Group1Score}", True, (255, 255, 255))
            G2 = font.render(f"{Group2Score}", True, (255, 255, 255))
            screen.blit(G1, (20, 10))
            screen.blit(G2, (width - G2.get_width() - 20, 10))

            # Show th winning group

            text = WinFont.render(f"Group {1 + np.argmax(np.array([Group1Score, Group2Score]))} won!", True,
                                  (236, 224, 60))
            text_rect = text.get_rect(center=(width / 2, height / 2))
            screen.blit(text, text_rect)

            pygame.display.flip()

            # Check for a quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    camera.stop()
                    pygame.quit()
                    exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        camera.stop()
                        pygame.quit()
                        exit()
            continue

        if flash > 0 and flash <= 3:
            flash += 1
        else:
            flash = 0

        # Compute the time passed in seconds
        elapsed_ticks = pygame.time.get_ticks() - gameTimer
        elapsed_seconds = int(elapsed_ticks / 1000)

        # Capture a frame from the camera
        image = camera.get_image()

        # Scale the image to fit the screen resolution
        image = pygame.transform.scale(image, (width, height))

        # Flip the Image
        image = pygame.transform.flip(image, True, False)

        # Draw the Cam image on the screen
        screen.blit(image, (0, 0))

        ## Optional!!! ## Draw split line on screen
        # pygame.draw.line(screen, (0,0,0,0), (width*0.5, 0), (width*0.5, height), width = 3)

        # Draw the target image on the screen
        screen.blit(target_image, (width * 0.4, height - width * 0.2))

        # Draw a progress bar
        if elapsed_ticks / (timePerStep * 1000) > 1:
            GroupTurn = 1 - GroupTurn  # Switch turn
            flash = 1

        if flash > 0:
            gameTimer = pygame.time.get_ticks()
            elapsed_seconds = 0
            alpha = 128  # Set the alpha value (transparency)
            rect_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            rect_surface.fill((255, 255, 255, alpha))  # Fill the surface with a transparent white color
            screen.blit(rect_surface, (0, 0))
        else:
            pygame.draw.rect(screen, (255, 0, 0, 128), (
            (width * 0.4, height - width * 0.2), ((width * 0.2 / (timePerStep * 1000)) * elapsed_ticks, 10)))

        if flash == 1:
            # Image2Numpy
            numpy_image = pygame.surfarray.array3d(image)
            mask_image = pygame.surfarray.array3d(target_image)
            if (1 - GroupTurn) == 1:
                ### Here we bring the image to the model ###
                G2CORR = model.cross(numpy_image, mask_image)
                ### Here we back to the mother fucker code ###

                if G2CORR > G1CORR:
                    Group2Score += 1
                else:
                    Group1Score += 1

                if Group1Score == 3:
                    Mode = "win"
                elif Group2Score == 3:
                    Mode = "win"

                G1CORR, G2CORR = 0, 0

                try:
                    index += 1
                    target_image = loadImage(images[index], width, height)
                except:
                    pass

            else:
                ### Here we bring the image to the model ###
                G1CORR = model.cross(numpy_image, target_image)
                ### Here we back to the mother fucker code ###

        if stop:
            gameTimer = pygame.time.get_ticks()
            elapsed_seconds = 0

        # Show the scores
        if GroupTurn == 0:
            G1 = font.render(f"{Group1Score}", True, (255, 255, 255))
            G2 = font.render(f"{Group2Score}", True, (91, 77, 77))
        else:
            G1 = font.render(f"{Group1Score}", True, (91, 77, 77))
            G2 = font.render(f"{Group2Score}", True, (255, 255, 255))

        screen.blit(G1, (25, 10))
        screen.blit(G2, (width - G2.get_width() - 25, 10))

        pygame.display.flip()

        # Check for a quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                camera.stop()
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    camera.stop()
                    pygame.quit()
                    exit()

                if event.key == pygame.K_r:  # Reset the timer on 'R' key press
                    gameTimer = pygame.time.get_ticks()
                    elapsed_seconds = 0

                if event.key == pygame.K_s:  # Reset and stop the timer on 'S' key press
                    stop = not stop

                if event.key == pygame.K_w:  # Reset and stop the timer on 'S' key press
                    Mode = "win"

except KeyboardInterrupt:
    camera.stop()
    pygame.quit()
