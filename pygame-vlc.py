import os
import sys
import vlc
import pygame


# Enable in Windows to use directx renderer instead of windib
#os.environ["SDL_VIDEODRIVER"] = "directx"

pygame.init()
screen = pygame.display.set_mode((800,600),pygame.RESIZABLE)
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


while player.get_state() != '':
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit(2)
		if event.type == pygame.KEYDOWN:
			print("OMG keydown!")
		if event.type == pygame.MOUSEBUTTONDOWN:
			print("we got a mouse button down!")
