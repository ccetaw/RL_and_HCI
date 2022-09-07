import numpy as np
import random
import json

def load_ui(filename: str):
    with open(filename, 'r') as file_to_read:
        temp = json.load(file_to_read)
    buttons = {}
    for i in range(len(temp)):
        buttons[i] = temp[f'{i}']
    return buttons

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class Button():
    def __init__(self, args) -> None:
        self.position = np.array(args['position']) # Left-top cornor of the button
        self.size = np.array(args['size']) # [width, height]
        self.id = args['id'] # string, the name of the button
        self.on = args['on'] # boolean, on/off
        self.type = args['type'] # central, mutual_exclu, normal 
        """
        central: 
            button turned from OFF to ON will turn on all normal buttons
                          from ON to OFF will turn off all normal buttons
        mutually exclusive: 
            only one button in this group can be ON. Turn on one button will turn off the others
            but buttons can be all OFF
        normal:
            turn it ON or OFF will not affect other buttons
        """

    def status(self):
        _status = {
            'position': self.position,
            'size': self.size,
            'id': self.id,
            'on': self.on,
            'type': self.type
        }
        return _status

class Interface():
    sample_id_pool = []
    for i in range(20):
        sample_id_pool.append(f'{hex(i)}') # Button name samples
    
    def __init__(self, config) -> None:
        self.screen_width = 960
        self.screen_height = 720
        self.grid_size = 80
        self.interval = 0 # unit: grid size
        self.margin_size = 40 # area where buttons cannot occupy
        self.buttons = {}
        self.button_group = {}

        self.n_buttons = config['n_buttons']
        self.random = config['random']

        grid_width = int((self.screen_width-2*self.margin_size)/self.grid_size) + 1
        grid_height = int((self.screen_height-2*self.margin_size)/self.grid_size) + 1
        self.grid = np.ones(shape=(grid_width, grid_height))

        if self.random:
            self.generate_random_ui()
        else:
            self.generate_sample_ui()

    def get_pixel(self, grid_point: np.array):
        pixel = np.array([self.margin_size, self.margin_size]) + grid_point * self.grid_size
        return pixel

    def is_available_grid_point(self, grid_position, grid_size):
        if np.all(grid_position + grid_size < self.grid.shape):
            to_occupy = self.grid[grid_position[0]:grid_position[0]+grid_size[0]+1, grid_position[1]:grid_position[1]+grid_size[1]+1]
            if np.array_equal(to_occupy, np.ones(np.shape(to_occupy))):
                return True
            else:
                return False
        else:
            return False
    
    def update_grid(self, button: Button):
        position = self.grid_button_position(button.position)
        size = self.grid_button_size(button.size)
        left = np.max([0, position[0]-1-self.interval])
        top = np.max([0, position[1]-1-self.interval])
        right = np.min([position[0]+size[0]+self.interval, self.grid.shape[0]-1])
        bottom = np.min([position[1]+size[1]+self.interval, self.grid.shape[1]-1])
        self.grid[left:right, top:bottom] = 0

    def classify_buttons(self):
        self.button_group = {
            'central': [],
            'normal': [],
            'mutual_exclu': []
        }
        for key, value in self.buttons.items():
            if value.type == 'central':
                self.button_group['central'].append(key)
            elif value.type == 'normal':
                self.button_group['normal'].append(key)
            elif value.type == 'mutual_exclu':
                self.button_group['mutual_exclu'].append(key)

    def generate_sample_ui(self): # Buttons have the same shape and are aligned 
        self.buttons.clear()
        self.grid = np.ones_like(self.grid)
        button_size = np.array([160, 160])
        for i in range(self.n_buttons):
            not_occupied = np.nonzero(self.grid)
            grid_button_position = np.array([not_occupied[0][0], not_occupied[1][0]])
            grid_button_size = self.grid_button_size(button_size)
            button_args = {
                'position': self.get_pixel(grid_button_position),
                'size': button_size,
                'id': self.sample_id_pool[i],
                'on': 0,
                'type': 'central' if i==0 else ('mutual_exclu' if i > self.n_buttons/2+1 else 'normal')
            }
            button = Button(button_args)
            for j in range(len(not_occupied[0])):
                if self.is_available_grid_point(grid_button_position, grid_button_size):
                    break
                else:
                    grid_button_position = np.array([not_occupied[0][j], not_occupied[1][j]])
            button.position = self.get_pixel(grid_button_position)
            self.buttons[i] = button
            self.update_grid(button)
        self.classify_buttons()

    def generate_random_ui(self): # Buttons have different shapes and randomly generated positions
        self.buttons.clear()
        self.grid = np.ones_like(self.grid)
        for i in range(self.n_buttons):
            not_occupied = np.nonzero(self.grid)
            button_size = np.random.randint(low=2, high=4, size=2) * 40
            grid_button_size = self.grid_button_size(button_size)
            sample_number = np.random.randint(low=0, high=np.size(not_occupied[0]))
            grid_button_position = np.array([not_occupied[0][sample_number], not_occupied[1][sample_number]])
            button_args = {
                'position': self.get_pixel(grid_button_position),
                'size': button_size,
                'id': self.sample_id_pool[i],
                'on': random.choice([0, 1]),
                'type': random.choice(['central', 'normal', 'mutual_exclu'])
            }
            button = Button(button_args)
            for _ in range(self.grid.size):
                if self.is_available_grid_point(grid_button_position, grid_button_size):
                    break
                else:
                    button_size = np.random.randint(low=2, high=4, size=2) * 40
                    grid_button_size = self.grid_button_size(button_size)
                    sample_number = np.random.randint(low=0, high=np.size(not_occupied[0]))
                    grid_button_position = np.array([not_occupied[0][sample_number], not_occupied[1][sample_number]])

            button.size = button_size
            button.position = self.get_pixel(grid_button_position)
            self.buttons[i] = button
            self.update_grid(button)

        self.classify_buttons()
        self.set_button_pattern(self.sample_possible_pattern())

    def generata_customized_ui(self, status: dict):
        self.buttons.clear()
        self.grid = np.ones_like(self.grid)
        flag = True
        for idx, button_arg in status.items():
            button = Button(button_arg)
            grid_button_position = self.grid_button_position(button.position)
            grid_button_size = self.grid_button_size(button.size)
            if self.is_available_grid_point(grid_button_position, grid_button_size): # Check if legal
                pass
            else:
                flag = False
            
            self.buttons[idx] = button
            self.update_grid(button) 
        self.classify_buttons()
        return flag
    
    def check_within_button(self, position):
        in_button = []
        for key, button in self.buttons.items():
            lefttop = self.normalized_button_position(button)
            size = self.normalized_button_size(button)
            if np.all(lefttop < position) and np.all(position < lefttop + size):
                in_button.append(key)
        return in_button

    def button_pattern(self):
        pattern = []
        for i in range(self.n_buttons):
            if self.buttons[i].on == 1:
                pattern.append(1)
            else:
                pattern.append(0)
        return pattern

    def set_button_pattern(self, pattern: np.array):
        for i in range(self.n_buttons):
            self.buttons[i].on = pattern[i]

    def press_button(self, idx: int):
        self.buttons[idx].on = 1 - self.buttons[idx].on
        if idx in self.button_group['central']:
            if self.buttons[idx].on == 1:
                for key in self.button_group['normal']:
                    self.buttons[key].on = 1
            else:
                for key in self.button_group['normal']:
                    self.buttons[key].on = 1
        elif idx in self.button_group['normal']:
            pass
        elif idx in self.button_group['mutual_exclu']:
            if self.buttons[idx].on == 1:
                for key in self.button_group['mutual_exclu']:
                    if key != idx:
                        self.buttons[key].on = 0
            else:
                pass
    
    def check_legal_pattern(self, pattern: np.array):
        to_check = pattern[self.button_group['mutual_exclu']]
        if np.sum(np.nonzero(to_check)[0]) > 1:
            return False
        else:
            return True

    def sample_possible_pattern(self):
        sample = np.random.randint(low=0, high=2, size=self.n_buttons)
        while not self.check_legal_pattern(sample):
            sample = np.random.randint(low=0, high=2, size=self.n_buttons)
        return sample

    def grid_button_size(self, size):
        return np.array(np.ceil(size / self.grid_size), dtype=int)

    def grid_button_position(self, position):
        return np.array(np.ceil(position / self.grid_size), dtype=int)

    def normalized_button_position(self, button: Button):
        return button.position / np.array([self.screen_width, self.screen_height])

    def normalized_button_size(self, button: Button):
        return button.size  / np.array([self.screen_width, self.screen_height])

    def show(self, show_grid=True):
        import pygame
        from pygame import gfxdraw

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        for _ in range(20): # Display 10 seconds
            canvas = pygame.Surface((self.screen_width, self.screen_height))
            font = pygame.font.Font('freesansbold.ttf', 16)
            canvas.fill((255, 255, 255))

            if show_grid:
                # Draw the margin
                pygame.draw.line(
                        canvas,
                        0,
                        (0, self.margin_size),
                        (self.screen_width, self.margin_size),
                        width=3,
                    )
                pygame.draw.line(
                        canvas,
                        0,
                        (0, self.screen_height - self.margin_size),
                        (self.screen_width, self.screen_height - self.margin_size),
                        width=3,
                    )
                pygame.draw.line(
                        canvas,
                        0,
                        (self.margin_size, 0),
                        (self.margin_size, self.screen_height),
                        width=3,
                    )
                pygame.draw.line(
                        canvas,
                        0,
                        (self.screen_width - self.margin_size, 0),
                        (self.screen_width - self.margin_size, self.screen_height),
                        width=3,
                    )

                # Draw the grid
                for y in range(self.grid.shape[1]):
                    pygame.draw.line(
                        canvas,
                        0,
                        (self.margin_size, self.margin_size + self.grid_size * y),
                        (self.screen_width - self.margin_size, self.margin_size + self.grid_size * y),
                        width=2,
                    )
                for x in range(self.grid.shape[0]):
                    pygame.draw.line(
                        canvas,
                        0,
                        (self.margin_size + self.grid_size * x, self.margin_size),
                        (self.margin_size + self.grid_size * x, self.screen_height - self.margin_size),
                        width=2,
                    )


            # Draw the buttons
            for button in self.buttons.values():
                if button.on:
                    pygame.draw.rect(
                        canvas,
                        (255, 165, 0),
                        pygame.Rect(
                            button.position,
                            (button.size[0], button.size[1]),
                        ),
                    )
                else:
                    pygame.draw.rect(
                        canvas,
                        (200, 200, 200),
                        pygame.Rect(
                            button.position,
                            (button.size[0], button.size[1]),
                        ),
                    )
                button_id = font.render(button.id, True, (150, 0, 200))
                button_type = font.render(button.type[0], True, (0, 178, 238))
                canvas.blit(button_id, button.position + 5)
                canvas.blit(button_type, [button.position[0] + button.size[0] - 15, button.position[1] + 5])
            
            

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(2)

        pygame.display.quit()
        pygame.quit()

    def status(self):
        _status = {}
        for key, value in self.buttons.items():
            _status[key] = value.status()
        return _status

    def save(self, filename: str):
        with open(filename, 'w') as file_to_write:
            json.dump(self.status(), file_to_write, indent=4, cls=NpEncoder)

if __name__ == '__main__':
    env_config = {
        'n_buttons': 4,
        'random': False
    }

    interface = Interface(env_config)
    interface.show()
    interface.press_button(0)
    interface.show()
    




