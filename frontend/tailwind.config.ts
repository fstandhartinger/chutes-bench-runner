import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Chutes brand colors
        ink: {
          100: "#F3F4F6",
          200: "#E5E7EB",
          300: "#D1D5DB",
          400: "#9CA3AF",
          500: "#1F2937",
          600: "#192231",
          700: "#142030",
          800: "#111726",
          900: "#0B0F14",
        },
        moss: "#63D297",
        papaya: "#FA5D19",
      },
      fontFamily: {
        sans: ["Tomato Grotesk", "Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;

























