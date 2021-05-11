#include "slime.cuh"

int main()
{
    sf::RenderWindow window(sf::VideoMode((int) WINDOW_WIDTH, (int) WINDOW_HEIGHT), "Slime Simulation", sf::Style::Fullscreen);

    uint numPixels = WINDOW_WIDTH*WINDOW_HEIGHT;
    uint numAgents = NUM_AGENTS;
    std::cout << "number of pixels in render: " << numPixels << std::endl;
    std::cout << "number of agents: " << numAgents << std::endl;

    struct Agent *agents;
    struct TrailMap *trailMap, *trailMapUpdated;
    uint8_t *pixels;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&agents, numAgents*sizeof(struct Agent));
    cudaMallocManaged(&trailMap, numPixels*sizeof(struct TrailMap));
    cudaMallocManaged(&trailMapUpdated, numPixels*sizeof(struct TrailMap));
    cudaMallocManaged(&pixels, numPixels*4*sizeof(uint8_t));

    for (int u = 0; u < WINDOW_WIDTH; u++)
    {
      for (int v = 0; v < WINDOW_HEIGHT; v++)
      {
        trailMap[v*WINDOW_WIDTH+u].x = u;
        trailMap[v*WINDOW_WIDTH+u].y = v;
        trailMap[v*WINDOW_WIDTH+u].val = 0;
        trailMap[v*WINDOW_WIDTH+u].sense = 0;
        trailMapUpdated[v*WINDOW_WIDTH+u].x = u;
        trailMapUpdated[v*WINDOW_WIDTH+u].y = v;
        trailMapUpdated[v*WINDOW_WIDTH+u].val = 0;
        trailMapUpdated[v*WINDOW_WIDTH+u].sense = 0;
      }
    }

    uint random;
    for (uint i = 0; i < numAgents; i++)
    {
      // agents[i].position.x = (uint) ((float) rand()/RAND_MAX * WINDOW_WIDTH*1)+WINDOW_WIDTH*0;
      // agents[i].position.y = (uint) ((float) rand()/RAND_MAX * WINDOW_HEIGHT*1)+WINDOW_HEIGHT*0;
      agents[i].position.x = WINDOW_WIDTH/2;
      agents[i].position.y = WINDOW_HEIGHT/2;
      agents[i].angle = ((float) rand()/RAND_MAX * 2 * M_PI);
    }

    sf::Image img;
    img.create(WINDOW_WIDTH, WINDOW_HEIGHT, sf::Color::White);
    sf::Color color;

    sf::Texture texture;
    sf::Sprite sprite;
    texture.create(WINDOW_WIDTH, WINDOW_HEIGHT);

    uint count = 0;
    clock_t t;
    t = clock();

    while (window.isOpen())
    {
      sf::Event event;
      while (window.pollEvent(event))
      {
        if (event.type == sf::Event::Closed) window.close();

        if (event.type == sf::Event::KeyPressed) 
        {
          if(event.key.code == sf::Keyboard::Escape) window.close();
        }
      }
      
      CUDA::wrapper(numAgents, agents, trailMap, trailMapUpdated, pixels);
      texture.update(pixels);
      sprite.setTexture(texture);
      window.clear(sf::Color::White);
      window.draw(sprite);
      window.display();
      count++;

      if((float) ((clock() - t) / CLOCKS_PER_SEC) >= 2.f)
      {
        std::cout << '\r' << "Average FPS: " << (float) count/2.f << std::flush;
        count = 0;
        t = clock();
      }
     }

    cudaFree(agents);
    cudaFree(trailMap);
    cudaFree(trailMapUpdated);
    cudaFree(pixels);

    return 0;
}