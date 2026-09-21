/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: {
          app: "#0B0D10",
          sidebar: "#0E1115",
          subtle: "#101318",
          surface: "#12161B",
          elevated: "#171C22",
          active: "#1C222A",
        },
        border: {
          default: "rgba(255, 255, 255, 0.06)",
          subtle: "rgba(255, 255, 255, 0.04)",
          emphasized: "rgba(255, 255, 255, 0.12)",
          focus: "#5BB8C4",
        },
        text: {
          primary: "#F3F5F7",
          secondary: "#A7AFBA",
          muted: "#6F7884",
          disabled: "#424852",
        },
        accent: {
          primary: "#5BB8C4",
          hover: "#71C7D1",
          subtle: "rgba(91, 184, 196, 0.12)",
        },
        status: {
          normal: "#55B98A",
          "normal-bg": "rgba(85, 185, 138, 0.12)",
          anomaly: "#E96B6B",
          "anomaly-bg": "rgba(233, 107, 107, 0.12)",
          review: "#D4A95B",
          "review-bg": "rgba(212, 169, 91, 0.12)",
          info: "#6C96D8",
          "info-bg": "rgba(108, 150, 216, 0.12)",
        },
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        mono: ["JetBrains Mono", "SFMono-Regular", "Menlo", "monospace"],
      },
      borderRadius: {
        sm: "4px",
        DEFAULT: "6px",
        md: "8px",
        lg: "10px",
      },
      boxShadow: {
        subtle: "0 1px 3px rgba(0, 0, 0, 0.3)",
        panel: "0 4px 20px rgba(0, 0, 0, 0.4)",
      },
    },
  },
  plugins: [],
};
