import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchTables, fetchPartitions, startInsightsJob, getJobStatus, type InsightsJob } from '@/lib/api';
import { useEffect } from 'react';

export const useTables = () => {
  return useQuery({
    queryKey: ['tables'],
    queryFn: fetchTables,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const usePartitions = (table: string | null) => {
  return useQuery({
    queryKey: ['partitions', table],
    queryFn: () => fetchPartitions(table!),
    enabled: !!table,
    staleTime: 5 * 60 * 1000,
  });
};

export const useStartInsights = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ table, partitions }: { table: string; partitions: Record<string, string[]> }) =>
      startInsightsJob(table, partitions),
    onSuccess: () => {
      // Invalidate job status queries to trigger polling
      queryClient.invalidateQueries({ queryKey: ['job-status'] });
    },
  });
};

export const useJobStatus = (jobId: string | null, enabled: boolean = true) => {
  const queryClient = useQueryClient();
  
  const query = useQuery({
    queryKey: ['job-status', jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: !!jobId && enabled,
    refetchInterval: (query) => {
      // Stop polling if job is complete or errored
      if (query.state.data?.status === 'complete' || query.state.data?.status === 'error') {
        return false;
      }
      return 2000; // Poll every 2 seconds
    },
    refetchIntervalInBackground: true,
  });

  // Stop polling when job completes
  useEffect(() => {
    if (query.data?.status === 'complete' || query.data?.status === 'error') {
      queryClient.setQueryData(['job-status', jobId], query.data);
    }
  }, [query.data?.status, queryClient, jobId]);

  return query;
};