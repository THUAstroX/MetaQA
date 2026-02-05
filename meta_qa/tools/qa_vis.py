"""
QA Visualization Module.

This module provides unified visualization tools for QA-enhanced scenario display:
- PIL-based rendering for GIF generation
- Pygame-based real-time display
- Camera grid layout (6 cameras in 3x2)

Classes:
    - QARenderer: PIL-based rendering functions for GIF frames
    - QADisplayItem: Data class for QA display items
    - QAInfoPanel: Pygame panel for QA annotations
    - ImagePanel: Pygame panel for camera images
    - IntegratedQAVisualizer: Combined MetaDrive + QA + Camera display

Camera Layout:
    [Front Left ] [  Front   ] [Front Right]
    [Back Left  ] [   Back   ] [Back Right ]
"""

import os
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# =============================================================================
# Constants
# =============================================================================

CAMERA_LAYOUT = [
    ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
]

CAMERA_LABELS = {
    'CAM_FRONT_LEFT': 'Front Left',
    'CAM_FRONT': 'Front',
    'CAM_FRONT_RIGHT': 'Front Right',
    'CAM_BACK_LEFT': 'Back Left',
    'CAM_BACK': 'Back',
    'CAM_BACK_RIGHT': 'Back Right',
}

# Color scheme
COLORS = {
    'background': (30, 30, 40),
    'header': (70, 130, 180),
    'text': (220, 220, 220),
    'question': (100, 200, 255),
    'answer': (100, 255, 150),
    'border': (80, 80, 100),
}

TYPE_COLORS = {
    "exist": (100, 200, 255),
    "count": (255, 200, 100),
    "object": (200, 150, 255),
    "status": (150, 255, 200),
    "comparison": (255, 150, 150),
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QADisplayItem:
    """A QA item formatted for display."""
    question: str
    answer: str
    template_type: str
    color: Tuple[int, int, int] = (255, 255, 255)


# =============================================================================
# PIL-based Rendering (for GIF generation)
# =============================================================================

class QARenderer:
    """
    PIL-based renderer for QA visualization.
    
    Used for generating static frames for GIF/video output.
    """
    
    def __init__(self, header_size: int = 16, normal_size: int = 14, small_size: int = 12):
        if not HAS_PIL:
            raise ImportError("PIL required for QARenderer")
        self.header_size = header_size
        self.normal_size = normal_size
        self.small_size = small_size
        self._fonts = None
    
    def _wrap_text(self, text: str, font: Any, max_width: int) -> List[str]:
        """Wrap text to fit within max_width pixels."""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            try:
                bbox = font.getbbox(test_line)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(test_line) * 7  # Fallback estimate
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [text]
    
    @property
    def fonts(self) -> Tuple[Any, Any, Any]:
        """Lazy load fonts."""
        if self._fonts is None:
            try:
                header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", self.header_size)
                normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.normal_size)
                small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.small_size)
            except:
                header = normal = small = ImageFont.load_default()
            self._fonts = (header, normal, small)
        return self._fonts
    
    def render_qa_panel(
        self,
        sample_qa: Optional[Any],
        description: str,
        width: int,
        height: int,
        frame_idx: int = 0,
        total_frames: int = 1,
    ) -> "Image.Image":
        """
        Render QA annotations panel.
        
        Args:
            sample_qa: SampleQAData object with qa_items
            description: Scene description
            width, height: Panel dimensions
            frame_idx: Current frame index
            total_frames: Total number of frames
            
        Returns:
            PIL Image of QA panel
        """
        img = Image.new('RGB', (width, height), COLORS['background'])
        draw = ImageDraw.Draw(img)
        header_font, font, small_font = self.fonts
        
        y = 10
        max_text_width = width - 30  # Leave margins
        
        # Header with frame info
        draw.text((10, y), "QA Annotations", fill=COLORS['header'], font=header_font)
        frame_text = f"Frame {frame_idx + 1}/{total_frames}"
        draw.text((width - 100, y), frame_text, fill=(150, 150, 150), font=small_font)
        y += 25
        
        # Scene description (with wrapping)
        if description:
            desc_lines = self._wrap_text(f"📍 {description}", small_font, max_text_width)
            for line in desc_lines[:2]:  # Max 2 lines for description
                draw.text((10, y), line, fill=(180, 180, 180), font=small_font)
                y += 14
            y += 4
        
        # Divider
        draw.line([(10, y), (width - 10, y)], fill=COLORS['border'], width=1)
        y += 10
        
        if sample_qa is None or len(sample_qa.qa_items) == 0:
            draw.text((10, y + 20), "No QA data for this keyframe", fill=(150, 150, 150), font=font)
            return img
        
        draw.text((10, y), f"Total QA items: {len(sample_qa.qa_items)}", fill=(200, 200, 200), font=small_font)
        y += 18
        
        # Show ALL QA items (dynamically calculate how many can fit)
        line_height = 14
        qa_item_height = line_height * 4 + 8  # type + question (2 lines max) + answer + spacing
        available_height = height - y - 20  # Leave bottom margin
        max_items = max(available_height // qa_item_height, 1)
        
        displayed = 0
        for i, qa in enumerate(sample_qa.qa_items):
            # Check if we have enough space for at least the type and one line of Q/A
            if y > height - 50:
                remaining = len(sample_qa.qa_items) - i
                if remaining > 0:
                    draw.text((10, y), f"... +{remaining} more", fill=(150, 150, 150), font=small_font)
                break
            
            type_color = TYPE_COLORS.get(qa.template_type, (200, 200, 200))
            draw.text((10, y), f"[{qa.template_type}]", fill=type_color, font=small_font)
            y += line_height
            
            # Question with wrapping (show all lines if space allows)
            q_lines = self._wrap_text(f"Q: {qa.question}", small_font, max_text_width - 10)
            max_q_lines = min(len(q_lines), 3)  # Max 3 lines per question
            for line in q_lines[:max_q_lines]:
                if y > height - 35:
                    break
                draw.text((15, y), line, fill=COLORS['question'], font=small_font)
                y += line_height
            
            # Answer with wrapping (usually short, max 2 lines)
            a_lines = self._wrap_text(f"A: {qa.answer}", small_font, max_text_width - 10)
            max_a_lines = min(len(a_lines), 2)
            for line in a_lines[:max_a_lines]:
                if y > height - 25:
                    break
                draw.text((15, y), line, fill=COLORS['answer'], font=small_font)
                y += line_height
            
            y += 8  # Spacing between QA items
            displayed += 1
        
        return img
    
    def render_camera_grid(
        self,
        image_paths: Dict[str, Optional[str]],
        width: int,
        height: int,
    ) -> "Image.Image":
        """
        Render 6-camera grid in 3x2 layout.
        
        Args:
            image_paths: Dict mapping camera name to image path
            width, height: Total grid dimensions
            
        Returns:
            PIL Image of camera grid
        """
        img = Image.new('RGB', (width, height), COLORS['background'])
        
        cols, rows = 3, 2
        cell_w, cell_h = width // cols, height // rows
        
        for row_idx, row_cameras in enumerate(CAMERA_LAYOUT):
            for col_idx, camera_name in enumerate(row_cameras):
                x, y = col_idx * cell_w, row_idx * cell_h
                image_path = image_paths.get(camera_name)
                cell_img = self._render_single_camera(image_path, camera_name, cell_w, cell_h)
                img.paste(cell_img, (x, y))
        
        # Draw grid lines
        draw = ImageDraw.Draw(img)
        for i in range(1, cols):
            draw.line([(i * cell_w, 0), (i * cell_w, height)], fill=COLORS['border'], width=1)
        draw.line([(0, cell_h), (width, cell_h)], fill=COLORS['border'], width=1)
        
        return img
    
    def _render_single_camera(
        self,
        image_path: Optional[str],
        camera_name: str,
        width: int,
        height: int,
    ) -> "Image.Image":
        """Render single camera cell."""
        img = Image.new('RGB', (width, height), COLORS['background'])
        draw = ImageDraw.Draw(img)
        _, font, small_font = self.fonts
        
        # Draw label
        label = CAMERA_LABELS.get(camera_name, camera_name)
        draw.text((5, 2), label, fill=(180, 180, 180), font=small_font)
        label_height = 20
        
        if image_path is None or not os.path.exists(image_path):
            draw.text((width // 2 - 15, height // 2), "N/A", fill=(100, 100, 100), font=font)
            return img
        
        try:
            cam_img = Image.open(image_path)
            target_w, target_h = width - 4, height - label_height - 4
            
            aspect = cam_img.width / cam_img.height
            if target_w / target_h > aspect:
                new_h, new_w = target_h, int(target_h * aspect)
            else:
                new_w, new_h = target_w, int(target_w / aspect)
            
            cam_img = cam_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x = (width - new_w) // 2
            y = label_height + (target_h - new_h) // 2 + 2
            img.paste(cam_img, (x, y))
        except Exception:
            draw.text((5, height // 2), "Error", fill=(255, 100, 100), font=small_font)
        
        return img


# =============================================================================
# Pygame-based Visualization (for real-time display)
# =============================================================================

class QAInfoPanel:
    """
    Pygame panel for displaying QA information.
    
    Shows QA questions/answers alongside MetaDrive visualization.
    """
    
    def __init__(self, width: int = 400, height: int = 600, font_size: int = 16, padding: int = 10):
        self.width = width
        self.height = height
        self.font_size = font_size
        self.padding = padding
        self._initialized = False
        self.font = self.small_font = self.header_font = self.surface = None
    
    def initialize(self):
        """Initialize pygame resources."""
        if not HAS_PYGAME:
            raise ImportError("pygame required for QAInfoPanel")
        
        pygame.font.init()
        self.font = pygame.font.Font(None, self.font_size)
        self.small_font = pygame.font.Font(None, self.font_size - 2)
        self.header_font = pygame.font.Font(None, self.font_size + 4)
        self.surface = pygame.Surface((self.width, self.height))
        self._initialized = True
    
    def render(
        self,
        scene_name: str,
        frame_idx: int,
        total_frames: int,
        qa_items: List[QADisplayItem],
        image_info: Optional[str] = None,
    ) -> "pygame.Surface":
        """Render the QA panel."""
        if not self._initialized:
            self.initialize()
        
        self.surface.fill(COLORS['background'])
        pygame.draw.rect(self.surface, COLORS['border'], (0, 0, self.width, self.height), 2)
        
        y = self.padding
        
        # Header
        header_surface = self.header_font.render(f"Scene: {scene_name}", True, COLORS['header'])
        self.surface.blit(header_surface, (self.padding, y))
        y += 30
        
        # Frame info
        frame_surface = self.font.render(f"Frame: {frame_idx + 1} / {total_frames}", True, COLORS['text'])
        self.surface.blit(frame_surface, (self.padding, y))
        y += 25
        
        # Divider
        pygame.draw.line(self.surface, COLORS['border'], (self.padding, y), (self.width - self.padding, y))
        y += 10
        
        # QA items
        qa_header_surface = self.font.render(f"QA Items ({len(qa_items)})", True, COLORS['header'])
        self.surface.blit(qa_header_surface, (self.padding, y))
        y += 25
        
        max_qa_height = self.height - y - 80
        
        for i, qa in enumerate(qa_items):
            if y - self.padding > max_qa_height:
                remaining_surface = self.small_font.render(f"... and {len(qa_items) - i} more", True, COLORS['text'])
                self.surface.blit(remaining_surface, (self.padding, y))
                break
            
            type_color = TYPE_COLORS.get(qa.template_type, COLORS['question'])
            type_surface = self.small_font.render(f"[{qa.template_type}]", True, type_color)
            self.surface.blit(type_surface, (self.padding, y))
            y += 18
            
            q_lines = self._wrap_text(f"Q: {qa.question}", self.width - 2 * self.padding)
            for line in q_lines:
                q_surface = self.small_font.render(line, True, COLORS['question'])
                self.surface.blit(q_surface, (self.padding + 10, y))
                y += 16
            
            a_surface = self.small_font.render(f"A: {qa.answer}", True, COLORS['answer'])
            self.surface.blit(a_surface, (self.padding + 10, y))
            y += 22
        
        # Image info at bottom
        if image_info:
            y = self.height - 40
            pygame.draw.line(self.surface, COLORS['border'], (self.padding, y), (self.width - self.padding, y))
            y += 10
            img_surface = self.small_font.render(f"📷 {image_info}", True, COLORS['text'])
            self.surface.blit(img_surface, (self.padding, y))
        
        return self.surface
    
    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text to fit width."""
        words = text.split()
        lines, current = [], []
        
        for word in words:
            test_line = ' '.join(current + [word])
            if self.small_font.size(test_line)[0] <= max_width:
                current.append(word)
            else:
                if current:
                    lines.append(' '.join(current))
                current = [word]
        
        if current:
            lines.append(' '.join(current))
        return lines[:3]


class ImagePanel:
    """Pygame panel for displaying camera images."""
    
    def __init__(self, width: int = 400, height: int = 300):
        self.width = width
        self.height = height
        self._initialized = False
        self.surface = None
        self._cache: Dict[str, Any] = {}
    
    def initialize(self):
        if not HAS_PYGAME:
            raise ImportError("pygame required for ImagePanel")
        self.surface = pygame.Surface((self.width, self.height))
        self._initialized = True
    
    def render(self, image_path: Optional[str], camera_name: str = "CAM_FRONT") -> "pygame.Surface":
        """Render camera image."""
        if not self._initialized:
            self.initialize()
        
        self.surface.fill((20, 20, 30))
        
        if image_path is None or not os.path.exists(image_path):
            font = pygame.font.Font(None, 24)
            text_surface = font.render("No image available", True, (150, 150, 150))
            text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2))
            self.surface.blit(text_surface, text_rect)
            return self.surface
        
        # Load and cache
        if image_path not in self._cache:
            try:
                if HAS_PIL:
                    pil_img = Image.open(image_path).resize((self.width - 20, self.height - 40), Image.Resampling.LANCZOS)
                    pygame_img = pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode)
                else:
                    pygame_img = pygame.transform.scale(pygame.image.load(image_path), (self.width - 20, self.height - 40))
                self._cache[image_path] = pygame_img
                if len(self._cache) > 100:
                    del self._cache[next(iter(self._cache))]
            except Exception:
                self._cache[image_path] = None
        
        if self._cache.get(image_path):
            self.surface.blit(self._cache[image_path], (10, 30))
        
        # Header
        font = pygame.font.Font(None, 20)
        self.surface.blit(font.render(camera_name, True, (200, 200, 200)), (10, 8))
        pygame.draw.rect(self.surface, COLORS['border'], (0, 0, self.width, self.height), 2)
        
        return self.surface
    
    def clear_cache(self):
        self._cache.clear()


class IntegratedQAVisualizer:
    """
    Combined MetaDrive + QA + Camera display.
    
    Layout:
        ┌───────────────────────┬─────────────┐
        │                       │  QA Panel   │
        │   MetaDrive Viewport  ├─────────────┤
        │                       │ Image Panel │
        └───────────────────────┴─────────────┘
    """
    
    def __init__(
        self,
        metadrive_width: int = 800,
        metadrive_height: int = 600,
        side_panel_width: int = 400,
    ):
        self.metadrive_width = metadrive_width
        self.metadrive_height = metadrive_height
        self.side_panel_width = side_panel_width
        
        self.total_width = metadrive_width + side_panel_width
        self.total_height = metadrive_height
        
        self.qa_panel = QAInfoPanel(width=side_panel_width, height=metadrive_height // 2)
        self.image_panel = ImagePanel(width=side_panel_width, height=metadrive_height // 2)
        
        self._initialized = False
        self.screen = None
    
    def initialize(self):
        if not HAS_PYGAME:
            raise ImportError("pygame required for visualization")
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.total_width, self.total_height))
        pygame.display.set_caption("MetaDrive + NuScenes-QA Visualization")
        
        self.qa_panel.initialize()
        self.image_panel.initialize()
        self._initialized = True
    
    def render(
        self,
        metadrive_surface: "pygame.Surface",
        scene_name: str,
        frame_idx: int,
        total_frames: int,
        qa_items: List[QADisplayItem],
        image_path: Optional[str] = None,
        camera_name: str = "CAM_FRONT",
    ) -> bool:
        """Render complete visualization. Returns False if quit requested."""
        if not self._initialized:
            self.initialize()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Scale MetaDrive surface if needed
        if metadrive_surface.get_size() != (self.metadrive_width, self.metadrive_height):
            metadrive_surface = pygame.transform.scale(metadrive_surface, (self.metadrive_width, self.metadrive_height))
        
        self.screen.blit(metadrive_surface, (0, 0))
        self.screen.blit(
            self.qa_panel.render(scene_name, frame_idx, total_frames, qa_items, 
                                os.path.basename(image_path) if image_path else None),
            (self.metadrive_width, 0)
        )
        self.screen.blit(
            self.image_panel.render(image_path, camera_name),
            (self.metadrive_width, self.metadrive_height // 2)
        )
        
        pygame.display.flip()
        return True
    
    def close(self):
        if self._initialized:
            pygame.quit()
            self._initialized = False


# =============================================================================
# Legacy compatibility exports
# =============================================================================

def load_fonts(header_size=16, normal_size=14, small_size=12):
    """Legacy function for loading fonts."""
    renderer = QARenderer(header_size, normal_size, small_size)
    return renderer.fonts

def render_qa_panel(sample_qa, description, width, height, frame_idx=0, total_frames=1, bg_color=(30, 30, 40)):
    """Legacy function for rendering QA panel."""
    renderer = QARenderer()
    return renderer.render_qa_panel(sample_qa, description, width, height, frame_idx, total_frames)

def render_camera_grid(image_paths, width, height, bg_color=(30, 30, 40), border_color=(60, 60, 80)):
    """Legacy function for rendering camera grid."""
    renderer = QARenderer()
    return renderer.render_camera_grid(image_paths, width, height)

def render_single_camera(image_path, camera_name, width, height, show_label=True, bg_color=(30, 30, 40)):
    """Legacy function for rendering single camera."""
    renderer = QARenderer()
    return renderer._render_single_camera(image_path, camera_name, width, height)
