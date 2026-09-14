import { create } from "zustand";

type UIState = { sidebarCollapsed: boolean; mobileOpen: boolean; toggleSidebar: () => void; setMobileOpen: (open: boolean) => void };
export const useUIStore = create<UIState>((set) => ({
  sidebarCollapsed: false,
  mobileOpen: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  setMobileOpen: (mobileOpen) => set({ mobileOpen }),
}));
