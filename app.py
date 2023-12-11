import pygame
import sys
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRYINC = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False

MODEL = load_model("mnist.h5")

LABELS = {
    0: "Zero", 1: "One",
    2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight",
    9: "Nine"
}

pygame.init()
FONT = pygame.font.Font(None, 36)

DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")

iswriting = False
number_xcord = []
number_ycord = []

image_cnt = 1

PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == pygame.MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == pygame.MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
            rect_min_Y, rect_max_Y = max(number_ycord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX, number_ycord[-1] + BOUNDRYINC)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_Y:rect_max_Y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite(f"image_{image_cnt}.png", img_arr)
                image_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, ((10, 10), (10, 10)), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255

                predicted_label = LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))]
                text_surface = FONT.render(predicted_label, True, RED, WHITE)
                text_rect = text_surface.get_rect(center=(rect_min_x + (rect_max_x - rect_min_x) // 2,
                                                          rect_min_Y + (rect_max_Y - rect_min_Y) // 2))

                DISPLAYSURF.blit(text_surface, text_rect)

            if event.type == pygame.KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill(BLACK)

    pygame.display.update()
    