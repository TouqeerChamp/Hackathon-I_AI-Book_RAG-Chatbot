/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Module 1: Foundations',
      items: ['intro', 'chapter-02-hardware'],
    },
    {
      type: 'category',
      label: 'Module 2: Core Skills',
      items: ['chapter-03-ros2', 'chapter-04-vision'],
    },
    {
      type: 'category',
      label: 'Module 3: Motion & Intelligence',
      items: ['chapter-05-control', 'chapter-06-ai-integration'],
    },
    {
      type: 'category',
      label: 'Module 4: Advanced Systems',
      items: ['chapter-07-navigation', 'chapter-08-ethics'],
    },
    'conclusion',
  ],
};

module.exports = sidebars;