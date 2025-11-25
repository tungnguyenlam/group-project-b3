# code/pipeline/Utils/MangaTypesetter.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

class MangaTypesetter:
    def __init__(self, font_families=['Comic Sans MS', 'Chalkboard SE', 'sans-serif']):
        self.font_props = fm.FontProperties(family=font_families)
        self.font_path = fm.findfont(self.font_props)
        self.text_color = (0, 0, 0)

    def render(self, original_image_rgb, bubbles):
        """
        Takes the original image and a list of bubbles (with 'translated_text').
        Returns the final numpy image with text typeset.
        """
        image_final = original_image_rgb.copy()
        erosion_kernel = np.ones((6, 6), np.uint8)

        # 1. Whitening (Inpainting) Step
        for bubble in bubbles:
            if not bubble.get('translated_text', '').strip(): continue
            
            # Use original_mask to have better overlay
            mask_to_use = bubble.get('original_mask', bubble['mask'])

            # Erode to clear old text without killing the border
            eroded_mask = cv2.erode(mask_to_use, erosion_kernel, iterations=1)
            contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_final, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        # 2. Text Rendering Step
        pil_image = Image.fromarray(image_final)
        draw = ImageDraw.Draw(pil_image)

        for bubble in bubbles:
            text = bubble.get('translated_text', '')
            if not text.strip(): continue
            
            self._fit_text_in_mask(draw, text, bubble['mask'])

        return np.array(pil_image)

    def _smart_wrap_text(self, draw, text, font, max_width_px):
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            w = draw.textbbox((0, 0), test_line, font=font)[2]
            if w <= max_width_px:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is wider than max width, force break
                    temp_word = ""
                    for char in word:
                        if draw.textbbox((0, 0), temp_word + char, font=font)[2] <= max_width_px:
                            temp_word += char
                        else:
                            lines.append(temp_word)
                            temp_word = char
                    current_line = [temp_word]
        if current_line:
            lines.append(' '.join(current_line))
        return lines

    def _check_mask_collision(self, text_mask, bubble_mask_crop):
        # Invert bubble: White = Wall
        bubble_wall = cv2.bitwise_not(bubble_mask_crop)
        # Intersection: Text touching Wall
        overlap = cv2.bitwise_and(text_mask, bubble_wall)
        return cv2.countNonZero(overlap) > 0

    def _fit_text_in_mask(self, draw, text, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Calculate Centroid
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2
        
        mask_crop = mask[y:y+h, x:x+w]
        
        font_size = min(h, w)
        min_font_size = 10
        best_font = None
        best_lines = []
        best_y_start = 0
        
        while font_size >= min_font_size:
            font = ImageFont.truetype(self.font_path, font_size)
            lines = self._smart_wrap_text(draw, text, font, w * 0.9)
            
            text_h = sum([draw.textbbox((0, 0), line, font=font)[3] for line in lines])
            text_w = max([draw.textbbox((0, 0), line, font=font)[2] for line in lines]) if lines else 0
            
            if text_h > h or text_w > w:
                font_size -= 2
                continue

            # Pixel Perfect Check
            text_canvas = np.zeros((h, w), dtype=np.uint8)
            pil_canvas = Image.fromarray(text_canvas)
            canvas_draw = ImageDraw.Draw(pil_canvas)
            
            curr_y = (cy - y) - (text_h / 2)
            rel_cx = cx - x
            
            for line in lines:
                line_w = draw.textbbox((0, 0), line, font=font)[2]
                line_h = draw.textbbox((0, 0), line, font=font)[3]
                canvas_draw.text((rel_cx - line_w/2, curr_y), line, font=font, fill=255)
                curr_y += line_h
            
            if not self._check_mask_collision(np.array(pil_canvas), mask_crop):
                best_font = font
                best_lines = lines
                best_y_start = cy - (text_h / 2)
                break
            
            font_size -= 2

        # Draw to main image
        if best_font:
            current_y = best_y_start
            for line in best_lines:
                line_w = draw.textbbox((0, 0), line, font=best_font)[2]
                line_h = draw.textbbox((0, 0), line, font=best_font)[3]
                draw.text((cx - line_w/2, current_y), line, font=best_font, fill=self.text_color)
                current_y += line_h