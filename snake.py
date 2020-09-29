import pygame
import numpy as np
import os
# import vlc
import random
# import multiprocessing
import neat


class Rectangle:
    food_imgs = ''
    head_imgs = ''
    body_imgs = ''
    tick = 0

    def __init__(self, pos, color=(0, 0, 0), direction=(1, 0)):
        self.dimension_x = Rectangle.dimension[0]
        self.dimension_y = Rectangle.dimension[1]
        self.pos = pos
        self.color = color
        self.direction = direction
        self.tick_body = 0

    def draw_body(self, surface):
        surface.blit(Rectangle.body_imgs[self.tick_body // 2],
                     (self.pos[0] * self.dimension_x, self.pos[1] * self.dimension_y))
        self.tick_body += 1
        if self.tick_body >= 28:
            self.tick_body = 0

    def draw_head(self, surface):
        surface.blit(Rectangle.head_imgs[self.tick], (self.pos[0] * self.dimension_x, self.pos[1] * self.dimension_y))
        self.tick += 1
        if self.tick >= 99:
            self.tick = 0

    def draw_food(self, surface):
        surface.blit(Rectangle.food_imgs[self.tick], (self.pos[0] * self.dimension_x, self.pos[1] * self.dimension_y))
        self.tick += 1
        if self.tick >= 49:
            self.tick = 0

    def draw(self, surface):
        rect = pygame.Rect((self.pos[0] * self.dimension_x, self.pos[1] * self.dimension_y),
                           (Rectangle.dimension[0], Rectangle.dimension[1]))
        pygame.draw.rect(surface, self.color, rect)

    def move(self, direction):
        self.direction = direction
        self.pos = (self.pos[0] + self.direction[0], self.pos[1] + self.direction[1])


class Snake(object):
    grid: object
    grid_size = (0, 0)

    def __init__(self, color, pos):
        grid_x, grid_y = self.grid_size
        self.body = []
        self.turns = {}
        self.color = color
        self.head = Rectangle(pos)
        self.body.append(self.head)
        self.direction = (0, 1)
        self.dead = False
        self.snack = Rectangle((random.randint(0, grid_x - 1), random.randint(0, grid_y - 1)))
        self.grid = np.zeros((grid_y, grid_x))
        self.grid[pos[1], pos[0]] = 10
        self.grid[self.snack.pos[1], self.snack.pos[1]] = 5
        self.grid = self.grid.tolist()

    def update(self, direction, grid_x, grid_y):
        self.grid = np.zeros((grid_y + 1, grid_x + 1))
        self.direction = direction
        self.turns[self.head.pos] = self.direction

        for i, c in enumerate(self.body):
            p = c.pos
            self.grid[c.pos[1], c.pos[0]] = 1
            if p == self.head.pos and i != 0:
                self.dead = True
            if p in self.turns:
                turn = self.turns[p]
                c.move((turn[0], turn[1]))
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.direction)
        if not ((0 <= self.head.pos[0] <= grid_x) and (0 <= self.head.pos[1] <= grid_y)):
            self.dead = True
        if self.head.pos == self.snack.pos:
            self.add_cube()
            self.snack = Rectangle((random.randint(0, grid_x), random.randint(0, grid_y)))
            self.grid[self.snack.pos[1], self.snack.pos[0]] = 5
            self.grid = self.grid.tolist()
            return True
        else:
            self.grid = self.grid.tolist()
            return False

    def reset(self, pos):
        self.head = Rectangle(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.direction = (0, 0)
        self.dead = False
        self.snack = Rectangle((5, 7))

    def add_cube(self):
        tail = self.body[-1]

        switch = {
            (1, 0): (Rectangle((tail.pos[0] - 1, tail.pos[1]))),
            (-1, 0): (Rectangle((tail.pos[0] + 1, tail.pos[1]))),
            (0, 1): (Rectangle((tail.pos[0], tail.pos[1] - 1))),
            (0, -1): (Rectangle((tail.pos[0], tail.pos[1] + 1))),
        }
        self.body.append(switch[tail.direction])
        """
        if dx == 1 and dy == 0:
            self.body.append(Rectangle((tail.pos[0] - 1, tail.pos[1])))
        elif dx == -1 and dy == 0:
            self.body.append(Rectangle((tail.pos[0] + 1, tail.pos[1])))
        elif dx == 0 and dy == 1:
            self.body.append(Rectangle((tail.pos[0], tail.pos[1] - 1)))
        elif dx == 0 and dy == -1:
            self.body.append(Rectangle((tail.pos[0], tail.pos[1] + 1)))
        """
        self.body[-1].direction = tail.direction

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw_head(surface)
            else:
                c.draw_body(surface)
        self.snack.draw_food(surface)

    def draw_simple(self, surface):
        for i, c in enumerate(self.body):
            c.draw(surface)
        self.snack.draw(surface)


def load_animation(directory_name, ):
    image_names = os.listdir(os.path.join('data', 'animations', directory_name))
    imgs = []
    for i, file_name in enumerate(image_names):
        imgs.append(pygame.image.load(os.path.join('data', 'animations', directory_name, file_name)))
        print(file_name)
    print(len(imgs))
    return imgs


def load_animation_with_scale(directory_name, scale):
    image_names = os.listdir(os.path.join('data', 'animations', directory_name))
    imgs = []
    for i, file_name in enumerate(image_names):
        imgs.append(
            pygame.transform.scale(pygame.image.load(os.path.join('data', 'animations', directory_name, file_name)),
                                   scale))
        print(file_name)
    print(len(imgs))
    return imgs


def draw_simple(surface, snakes, bg_imgs, tick):
    surface.fill((255, 0, 0))
    # surface.blit(bg_imgs[tick], (0, 0))
    for snake in snakes:
        snake.draw_simple(surface)
    pygame.display.flip()


def draw_complex(surface, snakes, bg_imgs, tick):
    surface.fill((255, 0, 0))
    surface.blit(bg_imgs[tick], (0, 0))
    for snake in snakes:
        snake.draw(surface)
    pygame.display.flip()


'''
def window():
    pygame.display.get_wm_info()
    movie = os.path.expanduser('4k-background-footage-ae-plugin-plexus.mp4')

    # Create instane of VLC and create reference to movie.
    vlcInstance = vlc.Instance('--input-repeat=999999')
    media = vlcInstance.media_new(movie)

    # Create new instance of vlc player
    player = vlcInstance.media_player_new()
    # Pass pygame window id to vlc player, so it can render its contents there.
    player.set_hwnd(pygame.display.get_wm_info()['window'])
    # Load movie into vlc player instance
    player.set_media(media)

    # Quit pygame mixer to allow vlc full access to audio device (REINIT AFTER MOVIE PLAYBACK IS FINISHED!)
    pygame.mixer.quit()

    # Start movie playback
    player.play()
'''


def eval_genomes(genomes, config):
    screen = pygame.display.get_surface()
    # setting Grid
    grid_factor = 1
    grid_x, grid_y = (int(16 * grid_factor), int(9 * grid_factor))

    # Normalising resolution to fit the grid
    w, h = screen.get_size()

    w = (w // grid_x) * grid_x
    h = (h // grid_y) * grid_y
    pygame.display.set_mode((w, h), )  # pygame.FULLSCREEN)
    # calculating a rectangle's dimension
    Rectangle.dimension = (w // grid_x, h // grid_y)
    Snake.grid_size = (grid_x, grid_y)

    # creating a snake instance, genomes, nets
    nets = []
    ge = []
    snakes = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        snakes.append(Snake(0, (grid_x // 2, grid_y // 2)))
        ge.append(genome)

    # Run until the user asks to quit
    running = True
    # Setting default input
    key_pressed = (0, 1)
    timer = 0
    # switch for handling inputs
    switch_inputs = {
        pygame.K_LEFT: (-1, 0),
        pygame.K_RIGHT: (1, 0),
        pygame.K_UP: (0, -1),
        pygame.K_DOWN: (0, 1),
    }
    # importing images
    # Rectangle.head_imgs = load_animation_with_scale('head', (Rectangle.dimension[0], Rectangle.dimension[1]))
    # Rectangle.body_imgs = load_animation_with_scale('body', (Rectangle.dimension[0], Rectangle.dimension[1]))
    # bg_imgs = load_animation('bg')
    bg_imgs = ''
    bg_imgs_len = len(bg_imgs)
    clock = pygame.time.Clock()
    tick = 0
    time = 0
    while running:
        # clock.tick(60)
        # inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    """
                try:
                    key_pressed = switch_inputs[event.key]
                except KeyError:
                    print('false Key')
                    """
        timer += 1
        if timer < 1000:
            timer += 1
            if len(snakes) > 0:
                for i, snake in enumerate(snakes):
                    if not snake.dead:
                        inputs = []
                        for parts in snake.grid:
                            for part in parts:
                                inputs.append(part)

                        output = nets[i].activate(inputs)
                        if output[0] > 0.5:
                            key_pressed = (-1, 0)
                        if output[1] > 0.5:
                            key_pressed = (1, 0)

                        if output[2] > 0.5:
                            key_pressed = (0, -1)

                        if output[3] > 0.5:
                            key_pressed = (0, 1)
                        ge[i].fitness += 0.01

                        if ge[i].fitness < -2:
                            nets.pop(i)
                            snakes.pop(i)
                            ge[i].fitness -= 1
                            ge.pop(i)

                        if snake.update(key_pressed, grid_x - 1, grid_y - 1):
                            ge[i].fitness += 10
                    else:
                        nets.pop(i)
                        snakes.pop(i)
                        ge[i].fitness -= 10
                        ge.pop(i)
            else:
                running = False
                break

        else:
            running = False
            break

        tick += 1
        if tick >= bg_imgs_len:
            tick = 0
        draw_simple(screen, snakes, bg_imgs, tick)
    # Done! Time to quit.
    # pygame.quit()


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play Snake.
    :param config_file: location of config file
    :return: None
    """
    # Initialising pygame
    pygame.init()
    pygame.display.set_mode()
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))
    # pe = neat.ThreadedEvaluator(4, eval_genomes)
    # winner = p.run(pe.evaluate, 300)
    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50000)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


def main_game():
    pygame.init()
    pygame.display.set_mode()
    screen = pygame.display.get_surface()
    # setting Grid
    grid_factor = 2
    grid_x, grid_y = (int(16 * grid_factor), int(9 * grid_factor))

    # Normalising resolution to fit the grid
    w, h = screen.get_size()

    w = (w // grid_x) * grid_x
    h = (h // grid_y) * grid_y
    pygame.display.set_mode((w, h), pygame.FULLSCREEN)
    # calculating a rectangle's dimension
    Rectangle.dimension = (w // grid_x, h // grid_y)
    Snake.grid_size = (grid_x, grid_y)

    # Creating Snake instance
    snakes = [Snake((100, 100, 100), (5, 5))]

    # Run until the user asks to quit
    running = True
    # Setting default input
    key_pressed = (0, 1)
    timer = 0
    # switch for handling inputs
    switch_inputs = {
        pygame.K_LEFT: (-1, 0),
        pygame.K_RIGHT: (1, 0),
        pygame.K_UP: (0, -1),
        pygame.K_DOWN: (0, 1),
    }
    # importing images
    Rectangle.head_imgs = load_animation_with_scale('head', (Rectangle.dimension[0], Rectangle.dimension[1]))
    Rectangle.body_imgs = load_animation_with_scale('body', (Rectangle.dimension[0], Rectangle.dimension[1]))
    Rectangle.food_imgs = load_animation_with_scale('food1', (Rectangle.dimension[0], Rectangle.dimension[1]))
    bg_imgs = load_animation('bg')
    bg_imgs_len = len(bg_imgs)
    clock = pygame.time.Clock()
    tick = 0
    timer = 0
    while running:
        clock.tick(60)
        # inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                try:
                    key_pressed = switch_inputs[event.key]
                except KeyError:
                    print('false Key')

        timer += 1
        if timer > 2:
            timer = 0
            for snake in snakes:
                if not snake.dead:
                    snake.update(key_pressed, grid_x - 1, grid_y - 1)
                else:
                    snake.reset((5, 5))

        tick += 1
        if tick >= bg_imgs_len:
            tick = 0
        draw_complex(screen, snakes, bg_imgs, tick)
    # Done! Time to quit.
    # pygame.quit()


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
    #main_game()
