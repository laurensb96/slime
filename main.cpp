#include "slime.cuh"

int main()
{
    sf::RenderWindow window(sf::VideoMode((int) WINDOW_WIDTH, (int) WINDOW_HEIGHT), "Slime Simulation");

    uint numPixels = WINDOW_WIDTH*WINDOW_HEIGHT;
    uint numAgents = NUM_AGENTS;
    std::cout << "number of pixels in render: " << numPixels << std::endl;
    std::cout << "number of agents: " << numAgents << std::endl;

    struct Agent *agentsGPU, *agents;
    struct TrailMap *trailMap;
    sf::Uint8 *pixels;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&agents, numAgents*sizeof(struct Agent));
    cudaMallocManaged(&trailMap, numPixels*sizeof(struct TrailMap));
    cudaMallocManaged(&pixels, numPixels*4*sizeof(sf::Uint8));

    for (int u = 0; u < WINDOW_WIDTH; u++)
    {
      for (int v = 0; v < WINDOW_HEIGHT; v++)
      {
        trailMap[v*WINDOW_WIDTH+u].x = u;
        trailMap[v*WINDOW_WIDTH+u].y = v;
        trailMap[v*WINDOW_WIDTH+u].val = 0;
        trailMap[v*WINDOW_WIDTH+u].sense = 0;
      }
    }

    uint random;
    for (uint i = 0; i < numAgents; i++)
    {
      agents[i].position.x = (uint) ((float) rand()/RAND_MAX * WINDOW_WIDTH*0.8)+WINDOW_WIDTH*0.1;
      agents[i].position.y = (uint) ((float) rand()/RAND_MAX * WINDOW_HEIGHT*0.8)+WINDOW_HEIGHT*0.1;
      agents[i].angle = (uint) ((float) rand()/RAND_MAX * 2 * M_PI);
    }

    sf::Image img;
    img.create(WINDOW_WIDTH, WINDOW_HEIGHT, sf::Color::White);
    sf::Color color;

    sf::Texture texture;
    sf::Sprite sprite;
    texture.create(WINDOW_WIDTH, WINDOW_HEIGHT);

    sf::Uint8 x;

    while (window.isOpen())
    {
      sf::Event event;
      while (window.pollEvent(event))
      {
        if (event.type == sf::Event::Closed) window.close();
      }
      
      CUDA::wrapper(numAgents, agents, trailMap, pixels);
      texture.update(pixels);
      sprite.setTexture(texture);
      window.clear(sf::Color::White);
      window.draw(sprite);
      window.display();
     }

    cudaFree(agents);
    cudaFree(trailMap);
    cudaFree(pixels);

    return 0;
}