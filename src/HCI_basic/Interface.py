import numpy as np

class Button():
    def __init__(self, args) -> None:
        self.position = np.array(args['position']) # Left-top cornor of the button
        self.size = np.array(args['size']) # [width, height]
        self.id = args['id'] # string, the name of the button

    def status(self):
        _status = {
            'position': self.position,
            'size': self.size,
            'id': self.id
        }
        return _status

class Interface():
    sample_id_pool = []
    for i in range(20):
        sample_id_pool.append(f'{hex(i)}') # Button name samples
    
    def __init__(self, config) -> None:
        self.screen_width = 800
        self.screen_height = 640
        self.grid_size = 40 
        self.interval = 0 # unit: grid size
        self.margin_size = 40 # area where buttons cannot occupy
        self.buttons = {}

        self.n_buttons = config['n_buttons']
        self.random = config['random']

        grid_width = int((self.screen_width-2*self.margin_size)/self.grid_size) + 1
        grid_height = int((self.screen_height-2*self.margin_size)/self.grid_size) + 1
        self.grid = np.ones(shape=(grid_width, grid_height))

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

    def generate_sample_ui(self): # Buttons have the same shape and are aligned 
        self.buttons.clear()
        self.grid = np.ones_like(self.grid)
        button_size = np.array([120, 120])
        for i in range(self.n_buttons):
            not_occupied = np.nonzero(self.grid)
            grid_button_position = np.array([not_occupied[0][0], not_occupied[1][0]])
            grid_button_size = self.grid_button_size(button_size)
            button_args = {
                'position': self.get_pixel(grid_button_position),
                'size': button_size,
                'id': self.sample_id_pool[i]
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


    def generate_random_ui(self): # Buttons have different shape and randomly generated position
        self.buttons.clear()
        self.grid = np.ones_like(self.grid)
        for i in range(self.n_buttons):
            not_occupied = np.nonzero(self.grid)
            button_size = np.random.randint(low=2, high=4, size=2) * self.grid_size
            grid_button_size = self.grid_button_size(button_size)
            sample_number = np.random.randint(low=0, high=np.size(not_occupied[0]))
            grid_button_position = np.array([not_occupied[0][sample_number], not_occupied[1][sample_number]])
            button_args = {
                'position': self.get_pixel(grid_button_position),
                'size': button_size,
                'id': self.sample_id_pool[i]
            }
            button = Button(button_args)
            for _ in range(100):
                if self.is_available_grid_point(grid_button_position, grid_button_size):
                    break
                else:
                    button_size = np.random.randint(low=2, high=4, size=2) * self.grid_size
                    grid_button_size = self.grid_button_size(button_size)
                    sample_number = np.random.randint(low=0, high=np.size(not_occupied[0]))
                    grid_button_position = np.array([not_occupied[0][sample_number], not_occupied[1][sample_number]])

            button.size = button_size
            button.position = self.get_pixel(grid_button_position)
            self.buttons[i] = button
            self.update_grid(button)



    def generata_customized_ui(self, buttons):
        self.buttons.clear()
        self.grid = np.ones_like(self.grid)

    def grid_button_size(self, size):
        return np.array(size / self.grid_size, dtype=int)

    def grid_button_position(self, position):
        return np.array(position / self.grid_size, dtype=int) 

    def normalized_button_position(self, button: Button):
        return button.position / np.array([self.screen_width, self.screen_height])

    def normalized_button_size(self, button: Button):
        return button.size  / np.array([self.screen_width, self.screen_height])

    def show(self, show_grid=False):
        import pygame
        from pygame import gfxdraw

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        for _ in range(20):
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
                pygame.draw.rect(
                    canvas,
                    (200, 200, 200),
                    pygame.Rect(
                        button.position,
                        (button.size[0], button.size[1]),
                    ),
                )
                text = font.render(button.id, True, (150, 0, 200))
                canvas.blit(text, button.position + 5)
            
            

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

if __name__ == '__main__':
    env_config = {
        'n_buttons': 10,
        'random': True
    }

    interface = Interface(env_config)
    interface.generate_sample_ui()
    interface.show(show_grid=False)
    print(interface.status())
        

    




